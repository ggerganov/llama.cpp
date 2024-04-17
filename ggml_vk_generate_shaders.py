#!/usr/bin/env python

import logging
import argparse
import asyncio
import os
import sys
from tempfile import gettempdir, NamedTemporaryFile

logger = logging.getLogger("ggml-vk-generate-shaders")

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
shader_f32_defines = """
#define QUANT_K 1
#define QUANT_R 1

#define A_TYPE float
"""
shader_f16_defines = """
#define QUANT_K 1
#define QUANT_R 1

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
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
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

# K-quants
shader_q2_K_defines = """
#define QUANT_K 256

struct block_q2_K
{
    uint8_t scales[QUANT_K/16];
    uint8_t qs[QUANT_K/4];
    f16vec2 d;
};

#define A_TYPE block_q2_K
"""
shader_q3_K_defines = """
#define QUANT_K 256

struct block_q3_K
{
    uint8_t hmask[QUANT_K/8];
    uint8_t qs[QUANT_K/4];
    uint8_t scales[12];
    float16_t d;
};

#define A_TYPE block_q3_K
"""
shader_q4_K_defines = """
#define QUANT_K 256

struct block_q4_K
{
    f16vec2 d;
    uint8_t scales[3*QUANT_K/64];
    uint8_t qs[QUANT_K/2];
};

#define A_TYPE block_q4_K
"""
shader_q5_K_defines = """
#define QUANT_K 256

struct block_q5_K
{
    f16vec2 d;
    uint8_t scales[12];
    uint8_t qh[QUANT_K/8];
    uint8_t qs[QUANT_K/2];
};

#define A_TYPE block_q5_K
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
shader_float_dequant_func = """
#define DEQUANT_FUNC vec2 v = vec2(ib, ib);  // data_a[ib], data_a[ib + 1]);
"""

shader_q4_0_dequant_func = """
#define DEQUANT_FUNC const float d = float(data_a[ib].d); \
const uint vui = uint(data_a[ib].qs[iqs]); \
vec2 v = vec2(vui & 0xF, vui >> 4); \
v = (v - 8.0f)*d;
"""

shader_q4_1_dequant_func = """
#define DEQUANT_FUNC const float d = float(data_a[ib].d); \
const float m = float(data_a[ib].m); \
const uint vui = uint(data_a[ib].qs[iqs]); \
vec2 v = vec2(vui & 0xF, vui >> 4); \
v = v*d + m;
"""

shader_q5_0_dequant_func = """
#define DEQUANT_FUNC const float d = float(data_a[ib].d); \
const uint uint_qh = uint(data_a[ib].qh[1]) << 16 | data_a[ib].qh[0]; \
const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10); \
const uint vui = uint(data_a[ib].qs[iqs]); \
vec2 v = vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = (v - 16.0f) * d;
"""

shader_q5_1_dequant_func = """
#define DEQUANT_FUNC const float d = float(data_a[ib].d); \
const float m = float(data_a[ib].m); \
const uint uint_qh = data_a[ib].qh; \
const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10); \
const uint vui = uint(data_a[ib].qs[iqs]); \
vec2 v = vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y); \
v = v*d + m;
"""

shader_q8_0_dequant_func = """
#define DEQUANT_FUNC const float d = float(data_a[ib].d); \
vec2 v = vec2(int(data_a[ib].qs[iqs]), int(data_a[ib].qs[iqs + 1])); \
v = v * d;
"""

# MULMAT

mulmat_head = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#ifndef LOAD_VEC_A
#define LOAD_VEC_A 1
#endif
#ifndef LOAD_VEC_B
#define LOAD_VEC_B 1
#endif
"""

mulmat_body1 = """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (push_constant) uniform parameter
{
    uint M;
    uint N;
    uint K;
    uint stride_a;
    uint stride_b;
    uint stride_d;
    uint k_split;

    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;
} p;

layout (constant_id = 1) const uint BM = 64;
layout (constant_id = 2) const uint BN = 64;
layout (constant_id = 3) const uint BK = 16;  // Assumed to be 32 if working with a quant
layout (constant_id = 4) const uint WM = 32;
layout (constant_id = 5) const uint WN = 32;
layout (constant_id = 6) const uint WMITER = 2;
layout (constant_id = 7) const uint TM = 4;
layout (constant_id = 8) const uint TN = 2;
layout (constant_id = 9) const uint WARP = 32;

shared FLOAT_TYPE buf_a[BM * (BK+1)];
shared FLOAT_TYPE buf_b[BN * (BK+1)];

void main() {
    const uint i13 = gl_GlobalInvocationID.z / p.ne12;
    const uint i12 = gl_GlobalInvocationID.z % p.ne12;

    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;

    const uint batch_idx_a = i03 * p.ne02 + i02;

    const uint blocks_m = (p.M + BM - 1) / BM;
    const uint ir = gl_WorkGroupID.x % blocks_m;
    const uint ik = gl_WorkGroupID.x / blocks_m;
    const uint ic = gl_WorkGroupID.y;

    const uint warp_i = gl_LocalInvocationID.x / WARP;
    const uint warp_r = warp_i % (BM / WM);
    const uint warp_c = warp_i / (BM / WM);

    const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    const uint WSUBM = WM / WMITER;
    const uint WSUBN = WN / WNITER;

    const uint tiw = gl_LocalInvocationID.x % WARP;
    const uint tiwr = tiw % (WSUBM / TM);
    const uint tiwc = tiw / (WSUBM / TM);

    const uint loadr_a = gl_LocalInvocationID.x % (BK / LOAD_VEC_A);
    const uint loadc_a = gl_LocalInvocationID.x / (BK / LOAD_VEC_A);
    const uint loadr_b = gl_LocalInvocationID.x % (BK / LOAD_VEC_B);
    const uint loadc_b = gl_LocalInvocationID.x / (BK / LOAD_VEC_B);

    const uint loadstride_a = gl_WorkGroupSize.x * LOAD_VEC_A / BK;
    const uint loadstride_b = gl_WorkGroupSize.x * LOAD_VEC_B / BK;

    const uint start_k = ik * p.k_split;
    const uint end_k = min(p.K, (ik + 1) * p.k_split);

    uint pos_a = (batch_idx_a * p.batch_stride_a + ir * BM * p.stride_a + start_k) / LOAD_VEC_A;
    uint pos_b = (gl_GlobalInvocationID.z * p.batch_stride_b + ic * BN * p.stride_b + start_k) / LOAD_VEC_B;

    float sums[WMITER * TM * WNITER * TN];
    FLOAT_TYPE cache_a[WMITER * TM];
    FLOAT_TYPE cache_b[WNITER * TN];

    [[unroll]] for (uint i = 0; i < WMITER*TM*WNITER*TN; i++) {
        sums[i] = 0.0f;
    }

    [[unroll]] for (uint block = start_k; block < end_k; block += BK) {
        [[unroll]] for (uint l = 0; l < BM; l += loadstride_a) {"""

mulmat_load_scalar = """
#if LOAD_VEC_A == 8
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;
            buf_a[buf_idx    ] = FLOAT_TYPE(data_a[idx][0].x);
            buf_a[buf_idx + 1] = FLOAT_TYPE(data_a[idx][0].y);
            buf_a[buf_idx + 2] = FLOAT_TYPE(data_a[idx][0].z);
            buf_a[buf_idx + 3] = FLOAT_TYPE(data_a[idx][0].w);
            buf_a[buf_idx + 4] = FLOAT_TYPE(data_a[idx][1].x);
            buf_a[buf_idx + 5] = FLOAT_TYPE(data_a[idx][1].y);
            buf_a[buf_idx + 6] = FLOAT_TYPE(data_a[idx][1].z);
            buf_a[buf_idx + 7] = FLOAT_TYPE(data_a[idx][1].w);
#elif LOAD_VEC_A == 4
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;
            buf_a[buf_idx    ] = FLOAT_TYPE(data_a[idx].x);
            buf_a[buf_idx + 1] = FLOAT_TYPE(data_a[idx].y);
            buf_a[buf_idx + 2] = FLOAT_TYPE(data_a[idx].z);
            buf_a[buf_idx + 3] = FLOAT_TYPE(data_a[idx].w);
#else
            if (ir * BM + loadc_a + l < p.M && block + loadr_a < end_k) {
                buf_a[(loadc_a + l) * (BK+1) + loadr_a] = FLOAT_TYPE(data_a[pos_a + (loadc_a + l) * p.stride_a + loadr_a]);
            } else {
                buf_a[(loadc_a + l) * (BK+1) + loadr_a] = FLOAT_TYPE(0.0f);
            }
#endif
"""

mulmat_load_q4_0 = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xF;

            const float d = float(data_a[ib].d);
            const uint vui = uint(data_a[ib].qs[iqs]);
            const vec2 v = (vec2(vui & 0xF, vui >> 4) - 8.0f) * d;

            buf_a[buf_idx     ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 16] = FLOAT_TYPE(v.y);"""

mulmat_load_q4_1 = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xF;

            const float d = float(data_a[ib].d);
            const float m = float(data_a[ib].m);
            const uint vui = uint(data_a[ib].qs[iqs]);
            const vec2 v = vec2(vui & 0xF, vui >> 4) * d + m;

            buf_a[buf_idx     ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 16] = FLOAT_TYPE(v.y);"""

mulmat_load_q5_0 = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xF;

            const float d = float(data_a[ib].d);
            const uint uint_qh = uint(data_a[ib].qh[1]) << 16 | data_a[ib].qh[0];
            const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
            const uint vui = uint(data_a[ib].qs[iqs]);
            const vec2 v = (vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) - 16.0f) * d;

            buf_a[buf_idx     ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 16] = FLOAT_TYPE(v.y);"""

mulmat_load_q5_1 = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xF;

            const float d = float(data_a[ib].d);
            const float m = float(data_a[ib].m);
            const uint uint_qh = data_a[ib].qh;
            const ivec2 qh = ivec2(((uint_qh >> iqs) << 4) & 0x10, (uint_qh >> (iqs + 12)) & 0x10);
            const uint vui = uint(data_a[ib].qs[iqs]);
            const vec2 v = vec2((vui & 0xF) | qh.x, (vui >> 4) | qh.y) * d + m;

            buf_a[buf_idx     ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 16] = FLOAT_TYPE(v.y);"""

mulmat_load_q8_0 = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 16;
            const uint iqs = (idx & 0xF) * 2;

            const float d = float(data_a[ib].d);
            const vec2 v = vec2(int(data_a[ib].qs[iqs]), int(data_a[ib].qs[iqs + 1])) * d;

            buf_a[buf_idx    ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 1] = FLOAT_TYPE(v.y);"""


mulmat_load_q2_K = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 128;                         // 2 values per idx
            const uint iqs = idx % 128;                        // 0..127

            const uint qsi = (iqs / 64) * 32 + (iqs % 16) * 2; // 0,2,4..30
            const uint scalesi = iqs / 8;                      // 0..15
            const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

            const uvec2 qs = uvec2(data_a[ib].qs[qsi], data_a[ib].qs[qsi + 1]);
            const uint scales = data_a[ib].scales[scalesi];
            const vec2 d = vec2(data_a[ib].d);

            const vec2 v = d.x * float(scales & 0xF) * vec2((qs >> qsshift) & 3) - d.y * float(scales >> 4);

            buf_a[buf_idx    ] = FLOAT_TYPE(v.x);
            buf_a[buf_idx + 1] = FLOAT_TYPE(v.y);"""

mulmat_load_q3_K = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint j = (iqs % 64) / 4;               // 0..3
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6
            const uint m = 1 << (4 * n + halfsplit);     // 1,2,4,8,16,32,64,128

            const int8_t us = int8_t(is <  4 ? (data_a[ib].scales[is-0] & 0xF) | (((data_a[ib].scales[is+8] >> 0) & 3) << 4) :
                                    is <  8 ? (data_a[ib].scales[is-0] & 0xF) | (((data_a[ib].scales[is+4] >> 2) & 3) << 4) :
                                    is < 12 ? (data_a[ib].scales[is-8] >>  4) | (((data_a[ib].scales[is+0] >> 4) & 3) << 4) :
                                            (data_a[ib].scales[is-8] >>  4) | (((data_a[ib].scales[is-4] >> 6) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            buf_a[buf_idx    ] = FLOAT_TYPE(dl * float(int8_t((data_a[ib].qs[qsi    ] >> qsshift) & 3) - (((data_a[ib].hmask[hmi    ] & m) != 0) ? 0 : 4)));
            buf_a[buf_idx + 1] = FLOAT_TYPE(dl * float(int8_t((data_a[ib].qs[qsi + 1] >> qsshift) & 3) - (((data_a[ib].hmask[hmi + 1] & m) != 0) ? 0 : 4)));"""

mulmat_load_q4_K = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 128;                 // 2 values per idx
            const uint iqs = idx % 128;                // 0..127

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

            const vec2 loadd = vec2(data_a[ib].d);

            uint8_t sc;
            uint8_t mbyte;
            if (is < 4) {
                sc    = uint8_t(data_a[ib].scales[is    ] & 63);
                mbyte = uint8_t(data_a[ib].scales[is + 4] & 63);
            } else {
                sc    = uint8_t((data_a[ib].scales[is + 4] & 0xF) | ((data_a[ib].scales[is - 4] >> 6) << 4));
                mbyte = uint8_t((data_a[ib].scales[is + 4] >>  4) | ((data_a[ib].scales[is    ] >> 6) << 4));
            }
            const float d = loadd.x * sc;
            const float m = loadd.y * mbyte;

            buf_a[buf_idx    ] = FLOAT_TYPE(d * float((data_a[ib].qs[qsi    ] >> (b * 4)) & 0xF) - m);
            buf_a[buf_idx + 1] = FLOAT_TYPE(d * float((data_a[ib].qs[qsi + 1] >> (b * 4)) & 0xF) - m);"""

mulmat_load_q5_K = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 128;                 // 2 values per idx
            const uint iqs = idx % 128;                // 0..127

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
            const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

            const uint8_t hm = uint8_t(1 << (iqs / 16));

            const vec2 loadd = vec2(data_a[ib].d);

            uint8_t sc;
            uint8_t mbyte;
            if (is < 4) {
                sc    = uint8_t(data_a[ib].scales[is    ] & 63);
                mbyte = uint8_t(data_a[ib].scales[is + 4] & 63);
            } else {
                sc    = uint8_t((data_a[ib].scales[is + 4] & 0xF) | ((data_a[ib].scales[is - 4] >> 6) << 4));
                mbyte = uint8_t((data_a[ib].scales[is + 4] >>  4) | ((data_a[ib].scales[is    ] >> 6) << 4));
            }
            const float d = loadd.x * sc;
            const float m = loadd.y * mbyte;

            buf_a[buf_idx    ] = FLOAT_TYPE(d * (float((data_a[ib].qs[qsi    ] >> (b * 4)) & 0xF) + float((data_a[ib].qh[qhi    ] & hm) != 0 ? 16 : 0)) - m);
            buf_a[buf_idx + 1] = FLOAT_TYPE(d * (float((data_a[ib].qs[qsi + 1] >> (b * 4)) & 0xF) + float((data_a[ib].qh[qhi + 1] & hm) != 0 ? 16 : 0)) - m);"""

mulmat_load_q6_K = """
            const uint idx = pos_a + (loadc_a + l) * p.stride_a / LOAD_VEC_A + loadr_a;
            const uint buf_idx = (loadc_a + l) * (BK+1) + loadr_a * LOAD_VEC_A;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint iqs = idx % 128;                 // 0..127

            const uint n = iqs / 64;                    // 0,1
            const uint b = (iqs % 64) / 32;             // 0,1
            const uint is_b = (iqs % 16) / 8;           // 0,1
            const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
            const uint is = 8 * n + qhshift + is_b;     // 0..15
            const uint qsi = n * 64 + (iqs % 32) * 2;   // 0,2,4..126
            const uint qhi = n * 32 + (iqs % 16) * 2;   // 0,2,4..62

            const float dscale = float(data_a[ib].d) * float(data_a[ib].scales[is]);

            buf_a[buf_idx    ] = FLOAT_TYPE(dscale * float(int8_t(((data_a[ib].ql[qsi    ] >> (b * 4)) & 0xF) | (((data_a[ib].qh[qhi    ] >> qhshift) & 3) << 4)) - 32));
            buf_a[buf_idx + 1] = FLOAT_TYPE(dscale * float(int8_t(((data_a[ib].ql[qsi + 1] >> (b * 4)) & 0xF) | (((data_a[ib].qh[qhi + 1] >> qhshift) & 3) << 4)) - 32));"""

mulmat_body2 = """
        }
        [[unroll]] for (uint l = 0; l < BN; l += loadstride_b) {
#if LOAD_VEC_B == 8
            const uint idx = pos_b + (loadc_b + l) * p.stride_b / LOAD_VEC_B + loadr_b;
            const uint buf_idx = (loadc_b + l) * (BK+1) + loadr_b * LOAD_VEC_B;
            buf_b[buf_idx + 0] = FLOAT_TYPE(data_b[idx][0].x);
            buf_b[buf_idx + 1] = FLOAT_TYPE(data_b[idx][0].y);
            buf_b[buf_idx + 2] = FLOAT_TYPE(data_b[idx][0].z);
            buf_b[buf_idx + 3] = FLOAT_TYPE(data_b[idx][0].w);
            buf_b[buf_idx + 4] = FLOAT_TYPE(data_b[idx][1].x);
            buf_b[buf_idx + 5] = FLOAT_TYPE(data_b[idx][1].y);
            buf_b[buf_idx + 6] = FLOAT_TYPE(data_b[idx][1].z);
            buf_b[buf_idx + 7] = FLOAT_TYPE(data_b[idx][1].w);
#elif LOAD_VEC_B == 4
            const uint idx = pos_b + (loadc_b + l) * p.stride_b / LOAD_VEC_B + loadr_b;
            const uint buf_idx = (loadc_b + l) * (BK+1) + loadr_b * LOAD_VEC_B;
            buf_b[buf_idx + 0] = FLOAT_TYPE(data_b[idx].x);
            buf_b[buf_idx + 1] = FLOAT_TYPE(data_b[idx].y);
            buf_b[buf_idx + 2] = FLOAT_TYPE(data_b[idx].z);
            buf_b[buf_idx + 3] = FLOAT_TYPE(data_b[idx].w);
#else
            if (ic * BN + loadc_b + l < p.N && block + loadr_b < end_k) {
                buf_b[(loadc_b + l) * (BK+1) + loadr_b] = FLOAT_TYPE(data_b[pos_b + (loadc_b + l) * p.stride_b + loadr_b]);
            } else {
                buf_b[(loadc_b + l) * (BK+1) + loadr_b] = FLOAT_TYPE(0.0f);
            }
#endif
        }

        barrier();

        pos_a += BK / LOAD_VEC_A;
        pos_b += BK / LOAD_VEC_B;

        for (uint i = 0; i < BK; i++) {
            // Load from shared into cache
            [[unroll]] for (uint wsir = 0; wsir < WMITER; wsir++) {
                [[unroll]] for (uint j = 0; j < TM; j++) {
                    cache_a[wsir * TM + j] = buf_a[(warp_r * WM + wsir * WSUBM + tiwr * TM + j) * (BK+1) + i];
                }
            }
            [[unroll]] for (uint wsic = 0; wsic < WNITER; wsic++) {
                [[unroll]] for (uint j = 0; j < TN; j++) {
                    cache_b[wsic * TN + j] = buf_b[(warp_c * WN + wsic * WSUBN + tiwc * TN + j) * (BK+1) + i];
                }
            }

            [[unroll]] for (uint wsic = 0; wsic < WNITER; wsic++) {
                [[unroll]] for (uint wsir = 0; wsir < WMITER; wsir++) {
                    [[unroll]] for (uint cc = 0; cc < TN; cc++) {
                        [[unroll]] for (uint cr = 0; cr < TM; cr++) {
                            sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr] += float(cache_a[wsir * TM + cr]) * float(cache_b[wsic * TN + cc]);
                        }
                    }
                }
            }
        }

        barrier();
    }

    const uint dr = ir * BM + warp_r * WM;
    const uint dc = ic * BN + warp_c * WN;

    const uint offsets = gl_GlobalInvocationID.z * p.batch_stride_d + ik * p.batch_stride_d * gl_NumWorkGroups.z;

    [[unroll]] for (uint wsic = 0; wsic < WNITER; wsic++) {
        [[unroll]] for (uint wsir = 0; wsir < WMITER; wsir++) {

            const uint dr_warp = dr + wsir * WSUBM + tiwr * TM;
            const uint dc_warp = dc + wsic * WSUBN + tiwc * TN;
            [[unroll]] for (uint cc = 0; cc < TN; cc++) {
                [[unroll]] for (uint cr = 0; cr < TM; cr++) {
                    if (dr_warp + cr < p.M && dc_warp + cc < p.N) {
                        data_d[offsets + (dc_warp + cc) * p.stride_d + dr_warp + cr] = D_TYPE(sums[(wsic * TN + cc) * (WMITER * TM) + wsir * TM + cr]);
                    }
                }
            }
        }
    }
}
"""

mulmat_split_k_reduce_src = """#version 450

#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {float data_a[];};
layout (binding = 1) writeonly buffer D {float data_d[];};

layout (push_constant) uniform parameter {
    uint ne;
    uint k_num;
} p;

void main() {
    const uint idx = gl_GlobalInvocationID.x;

    if (idx >= p.ne) {
        return;
    }

    float result = 0.0f;

    [[unroll]] for (uint i = 0; i < p.k_num; i++) {
        result += data_a[i * p.ne + idx];
    }

    data_d[idx] = result;
}
"""

# DEQUANT SHADER
dequant_head = """#version 450

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint M;
    uint K;
    uint stride_a;
    uint stride_b;
    uint nel;
} p;
"""

dequant_f32_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {float data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_GlobalInvocationID.x * 16;

    if (i >= p.nel) {
        return;
    }

    [[unroll]] for (uint l = 0; l < 16; l++) {
        data_b[i + l] = D_TYPE(data_a[i + l]);
    }
}
"""

dequant_q4_0_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {block_q4_0 data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid/32;
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
    if (ib >= p.nel / 32) {
        return;
    }

    const uint b_idx = 1024*i + 32*ir + 8*il;

    const float d = float(data_a[ib].d);
    const float dm = -8.0f * d;

    const uint q_idx = 8*il;

    [[unroll]] for (uint l = 0; l < 8; ++l) {
        data_b[b_idx + l +  0] = D_TYPE(d * (data_a[ib].qs[q_idx + l] & 0xF) + dm);
        data_b[b_idx + l + 16] = D_TYPE(d * (data_a[ib].qs[q_idx + l] >>  4) + dm);
    }
}
"""

dequant_q4_1_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {block_q4_1 data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid/32;
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
    if (ib >= p.nel / 32) {
        return;
    }

    const uint b_idx = 1024*i + 32*ir + 8*il;

    const float d = float(data_a[ib].d);
    const float m = float(data_a[ib].m);

    const uint q_idx = 8*il;

    [[unroll]] for (uint l = 0; l < 8; ++l) {
        data_b[b_idx + l +  0] = D_TYPE(d * (data_a[ib].qs[q_idx + l] & 0xF) + m);
        data_b[b_idx + l + 16] = D_TYPE(d * (data_a[ib].qs[q_idx + l] >>  4) + m);
    }
}
"""

dequant_q5_0_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {block_q5_0 data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid/32;
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
    if (ib >= p.nel / 32) {
        return;
    }

    const uint b_idx = 1024*i + 32*ir + 8*il;

    const float d = float(data_a[ib].d);
    const uint qh = uint(data_a[ib].qh[1]) << 16 | data_a[ib].qh[0];

    const uint q_idx = 8*il;

    [[unroll]] for (uint l = 0; l < 8; ++l) {
        const uint iqs = q_idx + l;
        const uint vui = uint(data_a[ib].qs[iqs]);
        data_b[b_idx + l +  0] = D_TYPE(d * (((vui & 0xF) | (((qh >> iqs) << 4) & 0x10)) - 16.0f));
        data_b[b_idx + l + 16] = D_TYPE(d * (((vui >>  4) | ((qh >> (iqs + 12)) & 0x10)) - 16.0f));
    }
}
"""

dequant_q5_1_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {block_q5_1 data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid/32;
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
    if (ib >= p.nel / 32) {
        return;
    }

    const uint b_idx = 1024*i + 32*ir + 8*il;

    const float d = float(data_a[ib].d);
    const float m = float(data_a[ib].m);
    const uint qh = data_a[ib].qh;

    const uint q_idx = 8*il;

    [[unroll]] for (uint l = 0; l < 8; ++l) {
        const uint iqs = q_idx + l;
        const uint vui = uint(data_a[ib].qs[iqs]);
        data_b[b_idx + l +  0] = D_TYPE(d * (((vui & 0xF) | (((qh >> iqs) << 4) & 0x10))) + m);
        data_b[b_idx + l + 16] = D_TYPE(d * (((vui >>  4) | ((qh >> (iqs + 12)) & 0x10))) + m);
    }
}
"""

dequant_q8_0_body = """
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {block_q8_0 data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;

    const uint tid = gl_LocalInvocationID.x % 64;
    const uint il  = tid/32;
    const uint ir  = tid%32;
    const uint ib = 32*i + ir;
    if (ib >= p.nel / 32) {
        return;
    }

    const uint b_idx = 1024*i + 32*ir + 16*il;

    const float d = float(data_a[ib].d);

    const uint q_idx = 16*il;

    [[unroll]] for (uint l = 0; l < 16; l += 2) {
        data_b[b_idx + l    ] = D_TYPE(d * data_a[ib].qs[q_idx + l    ]);
        data_b[b_idx + l + 1] = D_TYPE(d * data_a[ib].qs[q_idx + l + 1]);
    }
}
"""

# K-quants
dequant_q2_K_body = """
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.M * p.K / QUANT_K) {
            return;
        }

        const uint tid = gl_LocalInvocationID.x;
        const uint ip = tid / 32;
        const uint il = tid - 32 * ip;
        const uint is = 8 * ip + il / 16;

        const uint y_idx = i * QUANT_K + 128 * ip + il;

        const uint ql_idx = 32 * ip + il;
        const uint8_t qs = data_a[i].qs[32 * ip + il];

        FLOAT_TYPE dall = FLOAT_TYPE(data_a[i].d.x);
        FLOAT_TYPE dmin = FLOAT_TYPE(data_a[i].d.y);
        data_b[y_idx +  0] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is+0] & 0xF) * ((qs >> 0) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+0] >> 4));
        data_b[y_idx + 32] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is+2] & 0xF) * ((qs >> 2) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+2] >> 4));
        data_b[y_idx + 64] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is+4] & 0xF) * ((qs >> 4) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+4] >> 4));
        data_b[y_idx + 96] = D_TYPE(dall * FLOAT_TYPE((data_a[i].scales[is+6] & 0xF) * ((qs >> 6) & 3)) - dmin * FLOAT_TYPE(data_a[i].scales[is+6] >> 4));
    }
}
"""
dequant_q3_K_body = """
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = uint(gl_WorkGroupID.x * 256 + wgy);
        if (i >= p.M * p.K / QUANT_K) {
            return;
        }

        const uint r = gl_LocalInvocationID.x / 4;
        const uint tid = r / 2;
        const uint is0 = r % 2;
        const uint l0 = 16 * is0 + 4 * (gl_LocalInvocationID.x % 4);
        const uint n = tid / 4;
        const uint j = tid - 4*n;

        const uint8_t m = uint8_t(1 << (4*n + j));
        const uint is = 8*n + 2*j + is0;
        const uint shift = 2*j;

        const int8_t us = int8_t(is <  4 ? (data_a[i].scales[is-0] & 0xF) | (((data_a[i].scales[is+8] >> 0) & 3) << 4) :
                                 is <  8 ? (data_a[i].scales[is-0] & 0xF) | (((data_a[i].scales[is+4] >> 2) & 3) << 4) :
                                 is < 12 ? (data_a[i].scales[is-8] >>  4) | (((data_a[i].scales[is+0] >> 4) & 3) << 4) :
                                           (data_a[i].scales[is-8] >>  4) | (((data_a[i].scales[is-4] >> 6) & 3) << 4));
        const FLOAT_TYPE d_all = FLOAT_TYPE(data_a[i].d);
        const FLOAT_TYPE dl    = d_all * FLOAT_TYPE(us - 32);

        const uint y_idx = i * QUANT_K + 128 * n + 32 * j;
        const uint qs_idx = 32*n;

        for (uint l = l0; l < l0 + 4; ++l) {
            data_b[y_idx + l] = D_TYPE(dl * FLOAT_TYPE(int8_t((data_a[i].qs[qs_idx + l] >> shift) & 3) - (((data_a[i].hmask[l] & m) != 0) ? 0 : 4)));
        }
    }
}
"""
dequant_q4_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.M * p.K / QUANT_K) {
            return;
        }

        const uint tid = gl_LocalInvocationID.x;
        const uint il = tid / 8;
        const uint ir = tid % 8;
        const uint is = 2 * il;
        const uint n = 4;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[i].d.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[i].d.y);

        const uint y_idx = i * QUANT_K + 64 * il + n * ir;
        const uint qs_idx = 32*il + n * ir;

        uint8_t sc;
        uint8_t m;
        if (is < 4) {
            sc = uint8_t(data_a[i].scales[is] & 63);
            m  = uint8_t(data_a[i].scales[is + 4] & 63);
        } else {
            sc = uint8_t((data_a[i].scales[is + 4] & 0xF) | ((data_a[i].scales[is - 4] >> 6) << 4));
            m  = uint8_t((data_a[i].scales[is + 4] >>  4) | ((data_a[i].scales[is    ] >> 6) << 4));
        }
        const FLOAT_TYPE d1 = dall * sc;
        const FLOAT_TYPE m1 = dmin * m;

        if (is < 4) {
            sc = uint8_t(data_a[i].scales[is + 1] & 63);
            m  = uint8_t(data_a[i].scales[is + 5] & 63);
        } else {
            sc = uint8_t((data_a[i].scales[is + 5] & 0xF) | ((data_a[i].scales[is - 3] >> 6) << 4));
            m  = uint8_t((data_a[i].scales[is + 5] >>  4) | ((data_a[i].scales[is + 1] >> 6) << 4));
        }
        const FLOAT_TYPE d2 = dall * sc;
        const FLOAT_TYPE m2 = dmin * m;

        [[unroll]] for (uint l = 0; l < n; ++l) {
            data_b[y_idx + l     ] = D_TYPE(d1 * FLOAT_TYPE(data_a[i].qs[qs_idx + l] & 0xF) - m1);
            data_b[y_idx + l + 32] = D_TYPE(d2 * FLOAT_TYPE(data_a[i].qs[qs_idx + l] >>  4) - m2);
        }
    }
}
"""
dequant_q5_K_body = """
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.M * p.K / QUANT_K) {
            return;
        }

        const uint tid = gl_LocalInvocationID.x;
        const uint il = tid / 16;
        const uint ir = tid % 16;
        const uint is = 2 * il;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[i].d.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[i].d.y);

        const uint y_idx = i * QUANT_K + 64 * il + 2 * ir;
        const uint qs_idx = 32*il + 2 * ir;
        const uint qh_idx = 2 * ir;

        uint8_t sc;
        uint8_t m;
        if (is < 4) {
            sc = uint8_t(data_a[i].scales[is] & 63);
            m  = uint8_t(data_a[i].scales[is + 4] & 63);
        } else {
            sc = uint8_t((data_a[i].scales[is + 4] & 0xF) | ((data_a[i].scales[is - 4] >> 6) << 4));
            m  = uint8_t((data_a[i].scales[is + 4] >>  4) | ((data_a[i].scales[is    ] >> 6) << 4));
        }
        const FLOAT_TYPE d1 = dall * sc;
        const FLOAT_TYPE m1 = dmin * m;

        if (is < 4) {
            sc = uint8_t(data_a[i].scales[is + 1] & 63);
            m  = uint8_t(data_a[i].scales[is + 5] & 63);
        } else {
            sc = uint8_t((data_a[i].scales[is + 5] & 0xF) | ((data_a[i].scales[is - 3] >> 6) << 4));
            m  = uint8_t((data_a[i].scales[is + 5] >>  4) | ((data_a[i].scales[is + 1] >> 6) << 4));
        }
        const FLOAT_TYPE d2 = dall * sc;
        const FLOAT_TYPE m2 = dmin * m;

        const uint8_t hm1 = uint8_t(1 << (2 * il    ));
        const uint8_t hm2 = uint8_t(1 << (2 * il + 1));
        data_b[y_idx     ] = D_TYPE(d1 * FLOAT_TYPE((data_a[i].qs[qs_idx    ] & 0xF) + (((data_a[i].qh[qh_idx    ] & hm1) != 0) ? 16 : 0)) - m1);
        data_b[y_idx +  1] = D_TYPE(d1 * FLOAT_TYPE((data_a[i].qs[qs_idx + 1] & 0xF) + (((data_a[i].qh[qh_idx + 1] & hm1) != 0) ? 16 : 0)) - m1);
        data_b[y_idx + 32] = D_TYPE(d2 * FLOAT_TYPE((data_a[i].qs[qs_idx    ]  >> 4) + (((data_a[i].qh[qh_idx    ] & hm2) != 0) ? 16 : 0)) - m2);
        data_b[y_idx + 33] = D_TYPE(d2 * FLOAT_TYPE((data_a[i].qs[qs_idx + 1]  >> 4) + (((data_a[i].qh[qh_idx + 1] & hm2) != 0) ? 16 : 0)) - m2);
    }
}
"""
dequant_q6_K_body = """
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_b[];};

void main() {
    [[unroll]] for (uint wgy = 0; wgy < 256; wgy++) {
        const uint i = gl_WorkGroupID.x * 256 + wgy;
        if (i >= p.M * p.K / QUANT_K) {
            return;
        }
        const uint tid = gl_LocalInvocationID.x;
        const uint ip = tid / 32;
        const uint il = tid - 32 * ip;
        const uint is = 8 * ip + il / 16;

        const uint y_idx = i * QUANT_K + 128 * ip + il;

        const uint ql_idx = 64 * ip + il;
        const uint8_t qh = data_a[i].qh[32 * ip + il];

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[i].d);

        data_b[y_idx +  0] = D_TYPE(d * FLOAT_TYPE(data_a[i].scales[is + 0] * (int8_t((data_a[i].ql[ql_idx +  0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32)));
        data_b[y_idx + 32] = D_TYPE(d * FLOAT_TYPE(data_a[i].scales[is + 2] * (int8_t((data_a[i].ql[ql_idx + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32)));
        data_b[y_idx + 64] = D_TYPE(d * FLOAT_TYPE(data_a[i].scales[is + 4] * (int8_t((data_a[i].ql[ql_idx +  0] >>  4) | (((qh >> 4) & 3) << 4)) - 32)));
        data_b[y_idx + 96] = D_TYPE(d * FLOAT_TYPE(data_a[i].scales[is + 6] * (int8_t((data_a[i].ql[ql_idx + 32] >>  4) | (((qh >> 6) & 3) << 4)) - 32)));
    }
}
"""

# Mul Mat Vec
mul_mat_vec_head = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

layout (push_constant) uniform parameter
{
    uint ncols;
    uint b_offset;
    uint d_offset;
} p;
"""

mul_mat_vec_body = """
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

layout (constant_id = 0) const uint BLOCK_SIZE = 32;

shared FLOAT_TYPE tmp[BLOCK_SIZE];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    const uint y_offset = QUANT_R == 1 ? 1 : QUANT_K/2;

    tmp[tid] = FLOAT_TYPE(0.0f);

    [[unroll]] for (uint i = 0; i < p.ncols/BLOCK_SIZE; i += 2) {
        const uint col = i*BLOCK_SIZE + 2*tid;
        const uint ib = (row*p.ncols + col)/QUANT_K; // block index
        const uint iqs = (col%QUANT_K)/QUANT_R; // quant index
        const uint iybs = col - col%QUANT_K; // y block start index

        DEQUANT_FUNC

        // matrix multiplication
        tmp[tid] += FLOAT_TYPE(v.x) * FLOAT_TYPE(data_b[p.b_offset + iybs + iqs + 0]) +
                    FLOAT_TYPE(v.y) * FLOAT_TYPE(data_b[p.b_offset + iybs + iqs + y_offset]);
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""

# K-quants
mul_mat_vec_q2_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

shared FLOAT_TYPE tmp[32];

void main() {
    const uint row = gl_WorkGroupID.x;

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint ib0 = row*num_blocks_per_row;

    const uint tid = gl_LocalInvocationID.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const uint ix  = gl_LocalInvocationID.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const uint step = 16/K_QUANTS_PER_ITERATION;            // 16 or 8

    const uint v_im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_in = tid - step*v_im;                      // 0...15 or 0...7

    const uint l0 = K_QUANTS_PER_ITERATION*v_in;            // 0...15
    const uint q_offset = 32*v_im + l0;
    const uint s_offset = 8*v_im;
    const uint y_offset = 128*v_im + l0;

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    [[unroll]] for (uint i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const uint y_idx = i * QUANT_K + y_offset;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[ib0 + i].d.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[ib0 + i].d.y);

        FLOAT_TYPE sum1 = FLOAT_TYPE(0.0);
        FLOAT_TYPE sum2 = FLOAT_TYPE(0.0);
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += FLOAT_TYPE(data_b[p.b_offset + y_idx + l +  0]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 0] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l + 0] >> 0) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 16]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 1] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l +16] >> 0) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 32]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 2] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l + 0] >> 2) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 48]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 3] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l +16] >> 2) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 64]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 4] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l + 0] >> 4) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 80]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 5] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l +16] >> 4) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 96]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 6] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l + 0] >> 6) & 3)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l +112]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 7] & 0xF) * FLOAT_TYPE((data_a[ib0 + i].qs[q_offset + l +16] >> 6) & 3);
            sum2 += FLOAT_TYPE(data_b[p.b_offset + y_idx + l +  0]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 0] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 16]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 1] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 32]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 2] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 48]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 3] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 64]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 4] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 80]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 5] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 96]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 6] >> 4) & 0xF)
                  + FLOAT_TYPE(data_b[p.b_offset + y_idx + l +112]) * FLOAT_TYPE((data_a[ib0 + i].scales[s_offset + 7] >> 4) & 0xF);
        }
        tmp[16 * ix + tid] += dall * sum1 - dmin * sum2;
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""
mul_mat_vec_q3_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

shared FLOAT_TYPE tmp[32];

void main() {
    const uint row = gl_WorkGroupID.x;

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint ib0 = row*num_blocks_per_row;

    const uint tid = gl_LocalInvocationID.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const uint ix  = gl_LocalInvocationID.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const uint step = 16/K_QUANTS_PER_ITERATION;            // 16 or 8

    const uint v_im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_in = tid - step*v_im;                      // 0...15 or 0...7

    const uint8_t m = uint8_t(1 << (4 * v_im));

    const uint l0 = K_QUANTS_PER_ITERATION*v_in;            // 0...15
    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 128*v_im + l0;

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    const uint s_shift = 4 * v_im;

    [[unroll]] for (uint i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const uint y_idx = i * QUANT_K + y_offset;

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);

        FLOAT_TYPE sum = FLOAT_TYPE(0.0);
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum += FLOAT_TYPE(data_b[p.b_offset + y_idx + l +  0]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[0] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[ 8] >> (s_shift + 0) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l   ]     ) & 3) - (((data_a[ib0 + i].hmask[l0 + l   ] & (m << 0)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 32]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[2] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[10] >> (s_shift + 0) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l   ] >> 2) & 3) - (((data_a[ib0 + i].hmask[l0 + l   ] & (m << 1)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 64]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[4] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[ 8] >> (s_shift + 2) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l   ] >> 4) & 3) - (((data_a[ib0 + i].hmask[l0 + l   ] & (m << 2)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 96]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[6] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[10] >> (s_shift + 2) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l   ] >> 6) & 3) - (((data_a[ib0 + i].hmask[l0 + l   ] & (m << 3)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 16]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[1] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[ 9] >> (s_shift + 0) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l+16]     ) & 3) - (((data_a[ib0 + i].hmask[l0 + l+16] & (m << 0)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 48]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[3] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[11] >> (s_shift + 0) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l+16] >> 2) & 3) - (((data_a[ib0 + i].hmask[l0 + l+16] & (m << 1)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l + 80]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[5] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[ 9] >> (s_shift + 2) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l+16] >> 4) & 3) - (((data_a[ib0 + i].hmask[l0 + l+16] & (m << 2)) != 0) ? 0 : 4))
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l +112]) * FLOAT_TYPE(int8_t(((data_a[ib0 + i].scales[7] >> s_shift) & 0xF) | ((data_a[ib0 + i].scales[11] >> (s_shift + 2) & 0x3) << 4)) - 32) * FLOAT_TYPE(((data_a[ib0 + i].qs[q_offset + l+16] >> 6) & 3) - (((data_a[ib0 + i].hmask[l0 + l+16] & (m << 3)) != 0) ? 0 : 4));
        }
        tmp[16 * ix + tid] += d * sum;
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""
mul_mat_vec_q4_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

shared FLOAT_TYPE tmp[32];

void main() {
    const uint row = gl_WorkGroupID.x;

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint ib0 = row*num_blocks_per_row;

    const uint tid = gl_LocalInvocationID.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const uint ix  = gl_LocalInvocationID.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const uint step = 8/K_QUANTS_PER_ITERATION;             // 8 or 4

    const uint il = tid/step;                               // 0...3
    const uint ir = tid - step*il;                          // 0...7 or 0...3
    const uint n =  2 * K_QUANTS_PER_ITERATION;             // 2 or 4

    const uint v_im = il / 2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const uint v_in = il % 2;

    const uint l0 = n * (2 * ir + v_in);            // 0...15
    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 64*v_im + l0;

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    [[unroll]] for (uint i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const uint y1_idx = i * QUANT_K + y_offset;
        const uint y2_idx = y1_idx + 128;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[ib0 + i].d.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[ib0 + i].d.y);

        const uint8_t sc0 = uint8_t(  data_a[ib0 + i].scales[v_im * 2    ]       & 0x3f);
        const uint8_t sc1 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 1]       & 0x3f);
        const uint8_t sc2 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 4]       & 0x3f);
        const uint8_t sc3 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 5]       & 0x3f);
        const uint8_t sc4 = uint8_t(( data_a[ib0 + i].scales[v_im * 2 + 8]       & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2    ] & 0xc0) >> 2));
        const uint8_t sc5 = uint8_t(( data_a[ib0 + i].scales[v_im * 2 + 9]       & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 1] & 0xc0) >> 2));
        const uint8_t sc6 = uint8_t(((data_a[ib0 + i].scales[v_im * 2 + 8] >> 4) & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 4] & 0xc0) >> 2));
        const uint8_t sc7 = uint8_t(((data_a[ib0 + i].scales[v_im * 2 + 9] >> 4) & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 5] & 0xc0) >> 2));

#if K_QUANTS_PER_ITERATION == 2
        const uint8_t q4_0  = uint8_t(data_a[ib0 + i].qs[q_offset     ] & 0xf);
        const uint8_t q4_1  = uint8_t(data_a[ib0 + i].qs[q_offset +  1] & 0xf);
        const uint8_t q4_2  = uint8_t(data_a[ib0 + i].qs[q_offset +  2] & 0xf);
        const uint8_t q4_3  = uint8_t(data_a[ib0 + i].qs[q_offset +  3] & 0xf);
        const uint8_t q4_4  = uint8_t(data_a[ib0 + i].qs[q_offset     ]  >> 4);
        const uint8_t q4_5  = uint8_t(data_a[ib0 + i].qs[q_offset +  1]  >> 4);
        const uint8_t q4_6  = uint8_t(data_a[ib0 + i].qs[q_offset +  2]  >> 4);
        const uint8_t q4_7  = uint8_t(data_a[ib0 + i].qs[q_offset +  3]  >> 4);
        const uint8_t q4_8  = uint8_t(data_a[ib0 + i].qs[q_offset + 64] & 0xf);
        const uint8_t q4_9  = uint8_t(data_a[ib0 + i].qs[q_offset + 65] & 0xf);
        const uint8_t q4_10 = uint8_t(data_a[ib0 + i].qs[q_offset + 66] & 0xf);
        const uint8_t q4_11 = uint8_t(data_a[ib0 + i].qs[q_offset + 67] & 0xf);
        const uint8_t q4_12 = uint8_t(data_a[ib0 + i].qs[q_offset + 64]  >> 4);
        const uint8_t q4_13 = uint8_t(data_a[ib0 + i].qs[q_offset + 65]  >> 4);
        const uint8_t q4_14 = uint8_t(data_a[ib0 + i].qs[q_offset + 66]  >> 4);
        const uint8_t q4_15 = uint8_t(data_a[ib0 + i].qs[q_offset + 67]  >> 4);

        const FLOAT_TYPE sx = FLOAT_TYPE(data_b[p.b_offset + y1_idx] * q4_0 + data_b[p.b_offset + y1_idx + 1] * q4_1 + data_b[p.b_offset + y1_idx + 2] * q4_2 + data_b[p.b_offset + y1_idx + 3] * q4_3);
        const FLOAT_TYPE sy = FLOAT_TYPE(data_b[p.b_offset + y1_idx + 32] * q4_4 + data_b[p.b_offset + y1_idx + 33] * q4_5 + data_b[p.b_offset + y1_idx + 34] * q4_6 + data_b[p.b_offset + y1_idx + 35] * q4_7);
        const FLOAT_TYPE sz = FLOAT_TYPE(data_b[p.b_offset + y2_idx] * q4_8 + data_b[p.b_offset + y2_idx + 1] * q4_9 + data_b[p.b_offset + y2_idx + 2] * q4_10 + data_b[p.b_offset + y2_idx + 3] * q4_11);
        const FLOAT_TYPE sw = FLOAT_TYPE(data_b[p.b_offset + y2_idx + 32] * q4_12 + data_b[p.b_offset + y2_idx + 33] * q4_13 + data_b[p.b_offset + y2_idx + 34] * q4_14 + data_b[p.b_offset + y2_idx + 35] * q4_15);
        const FLOAT_TYPE smin = FLOAT_TYPE(
            data_b[p.b_offset + y1_idx    ] * sc2 + data_b[p.b_offset + y1_idx + 32] * sc3 + data_b[p.b_offset + y2_idx    ] * sc6 + data_b[p.b_offset + y2_idx + 32] * sc7
          + data_b[p.b_offset + y1_idx + 1] * sc2 + data_b[p.b_offset + y1_idx + 33] * sc3 + data_b[p.b_offset + y2_idx + 1] * sc6 + data_b[p.b_offset + y2_idx + 33] * sc7
          + data_b[p.b_offset + y1_idx + 2] * sc2 + data_b[p.b_offset + y1_idx + 34] * sc3 + data_b[p.b_offset + y2_idx + 2] * sc6 + data_b[p.b_offset + y2_idx + 34] * sc7
          + data_b[p.b_offset + y1_idx + 3] * sc2 + data_b[p.b_offset + y1_idx + 35] * sc3 + data_b[p.b_offset + y2_idx + 3] * sc6 + data_b[p.b_offset + y2_idx + 35] * sc7
        );
        tmp[16 * ix + tid] += FLOAT_TYPE(dall * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dmin * smin);
#else
        const uint8_t q4_0 = uint8_t(data_a[ib0 + i].qs[q_offset     ] & 0xf);
        const uint8_t q4_1 = uint8_t(data_a[ib0 + i].qs[q_offset +  1] & 0xf);
        const uint8_t q4_2 = uint8_t(data_a[ib0 + i].qs[q_offset     ]  >> 4);
        const uint8_t q4_3 = uint8_t(data_a[ib0 + i].qs[q_offset +  1]  >> 4);
        const uint8_t q4_4 = uint8_t(data_a[ib0 + i].qs[q_offset + 64] & 0xf);
        const uint8_t q4_5 = uint8_t(data_a[ib0 + i].qs[q_offset + 65] & 0xf);
        const uint8_t q4_6 = uint8_t(data_a[ib0 + i].qs[q_offset + 64]  >> 4);
        const uint8_t q4_7 = uint8_t(data_a[ib0 + i].qs[q_offset + 65]  >> 4);

        const FLOAT_TYPE sx = FLOAT_TYPE(data_b[p.b_offset + y1_idx     ] * q4_0  + data_b[p.b_offset + y1_idx +  1] * q4_1);
        const FLOAT_TYPE sy = FLOAT_TYPE(data_b[p.b_offset + y1_idx + 32] * q4_2  + data_b[p.b_offset + y1_idx + 33] * q4_3);
        const FLOAT_TYPE sz = FLOAT_TYPE(data_b[p.b_offset + y2_idx     ] * q4_4  + data_b[p.b_offset + y2_idx +  1] * q4_5);
        const FLOAT_TYPE sw = FLOAT_TYPE(data_b[p.b_offset + y2_idx + 32] * q4_6 + data_b[p.b_offset + y2_idx + 33] * q4_7);
        const FLOAT_TYPE smin = FLOAT_TYPE(
            data_b[p.b_offset + y1_idx] * sc2 + data_b[p.b_offset + y1_idx + 32] * sc3 + data_b[p.b_offset + y2_idx] * sc6 + data_b[p.b_offset + y2_idx + 32] * sc7
          + data_b[p.b_offset + y1_idx + 1] * sc2 + data_b[p.b_offset + y1_idx + 33] * sc3 + data_b[p.b_offset + y2_idx + 1] * sc6 + data_b[p.b_offset + y2_idx + 33] * sc7
        );

        tmp[16 * ix + tid] += FLOAT_TYPE(dall * (sx * FLOAT_TYPE(data_a[ib0 + i].scales[v_im] & 0x3f) + sy * FLOAT_TYPE(data_a[ib0 + i].scales[v_im + 1] & 0x3f) + sz * FLOAT_TYPE((data_a[ib0 + i].scales[v_im + 4] & 0x0f) | ((data_a[ib0 + i].scales[v_im] & 0xc0) >> 2)) + sw * FLOAT_TYPE((data_a[ib0 + i].scales[v_im + 5] & 0x0f) | ((data_a[ib0 + i].scales[v_im + 1] & 0xc0) >> 2))) - dmin * smin);
#endif
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""
mul_mat_vec_q5_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

shared FLOAT_TYPE tmp[32];

void main() {
    const uint row = gl_WorkGroupID.x;

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint ib0 = row*num_blocks_per_row;

    const uint tid = gl_LocalInvocationID.x/2;  // 0...31 or 0...16
    const uint ix  = gl_LocalInvocationID.x%2;  // 0 or 0, 1

    const uint il = tid/4;                           // 0...3
    const uint ir = tid - 4*il;                      // 0...7 or 0...3

    const uint v_im = il / 2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const uint v_in = il % 2;

    const uint l0 = 4*ir + 2*v_in;                   // 0...15
    const uint q_offset = 32*v_im + l0;
    const uint y_offset = 64*v_im + l0;

    const uint8_t hm1 = uint8_t(1 << (2*v_im));
    const uint8_t hm2 = uint8_t(hm1 << 4);

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    [[unroll]] for (uint i = ix; i < num_blocks_per_row; i += 2) {
        const uint y1_idx = i * QUANT_K + y_offset;
        const uint y2_idx = y1_idx + 128;

        const FLOAT_TYPE dall = FLOAT_TYPE(data_a[ib0 + i].d.x);
        const FLOAT_TYPE dmin = FLOAT_TYPE(data_a[ib0 + i].d.y);

        const uint8_t sc0 = uint8_t(  data_a[ib0 + i].scales[v_im * 2    ]       & 0x3f);
        const uint8_t sc1 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 1]       & 0x3f);
        const uint8_t sc2 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 4]       & 0x3f);
        const uint8_t sc3 = uint8_t(  data_a[ib0 + i].scales[v_im * 2 + 5]       & 0x3f);
        const uint8_t sc4 = uint8_t(( data_a[ib0 + i].scales[v_im * 2 + 8]       & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2    ] & 0xc0) >> 2));
        const uint8_t sc5 = uint8_t(( data_a[ib0 + i].scales[v_im * 2 + 9]       & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 1] & 0xc0) >> 2));
        const uint8_t sc6 = uint8_t(((data_a[ib0 + i].scales[v_im * 2 + 8] >> 4) & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 4] & 0xc0) >> 2));
        const uint8_t sc7 = uint8_t(((data_a[ib0 + i].scales[v_im * 2 + 9] >> 4) & 0x0f) | ((data_a[ib0 + i].scales[v_im * 2 + 5] & 0xc0) >> 2));

        const uint8_t q4_0  = uint8_t(data_a[ib0 + i].qs[q_offset     ] & 0xf);
        const uint8_t q4_1  = uint8_t(data_a[ib0 + i].qs[q_offset +  1] & 0xf);
        const uint8_t q4_2  = uint8_t(data_a[ib0 + i].qs[q_offset + 16] & 0xf);
        const uint8_t q4_3  = uint8_t(data_a[ib0 + i].qs[q_offset + 17] & 0xf);
        const uint8_t q4_4  = uint8_t(data_a[ib0 + i].qs[q_offset     ]  >> 4);
        const uint8_t q4_5  = uint8_t(data_a[ib0 + i].qs[q_offset +  1]  >> 4);
        const uint8_t q4_6  = uint8_t(data_a[ib0 + i].qs[q_offset + 16]  >> 4);
        const uint8_t q4_7  = uint8_t(data_a[ib0 + i].qs[q_offset + 17]  >> 4);
        const uint8_t q4_8  = uint8_t(data_a[ib0 + i].qs[q_offset + 64] & 0xf);
        const uint8_t q4_9  = uint8_t(data_a[ib0 + i].qs[q_offset + 65] & 0xf);
        const uint8_t q4_10 = uint8_t(data_a[ib0 + i].qs[q_offset + 80] & 0xf);
        const uint8_t q4_11 = uint8_t(data_a[ib0 + i].qs[q_offset + 81] & 0xf);
        const uint8_t q4_12 = uint8_t(data_a[ib0 + i].qs[q_offset + 64]  >> 4);
        const uint8_t q4_13 = uint8_t(data_a[ib0 + i].qs[q_offset + 65]  >> 4);
        const uint8_t q4_14 = uint8_t(data_a[ib0 + i].qs[q_offset + 80]  >> 4);
        const uint8_t q4_15 = uint8_t(data_a[ib0 + i].qs[q_offset + 81]  >> 4);

        const FLOAT_TYPE sx = FLOAT_TYPE(
            data_b[p.b_offset + y1_idx     ] * (q4_0 + (((data_a[ib0 + i].qh[l0     ] & hm1) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx +  1] * (q4_1 + (((data_a[ib0 + i].qh[l0 +  1] & hm1) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx + 16] * (q4_2 + (((data_a[ib0 + i].qh[l0 + 16] & hm1) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx + 17] * (q4_3 + (((data_a[ib0 + i].qh[l0 + 17] & hm1) != 0) ? 16 : 0))
        );
        const FLOAT_TYPE sy = FLOAT_TYPE(
            data_b[p.b_offset + y1_idx + 32] * (q4_4 + (((data_a[ib0 + i].qh[l0     ] & (hm1 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx + 33] * (q4_5 + (((data_a[ib0 + i].qh[l0 +  1] & (hm1 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx + 48] * (q4_6 + (((data_a[ib0 + i].qh[l0 + 16] & (hm1 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y1_idx + 49] * (q4_7 + (((data_a[ib0 + i].qh[l0 + 17] & (hm1 << 1)) != 0) ? 16 : 0))
        );
        const FLOAT_TYPE sz = FLOAT_TYPE(
            data_b[p.b_offset + y2_idx     ] * (q4_8  + (((data_a[ib0 + i].qh[l0     ] & hm2) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx +  1] * (q4_9  + (((data_a[ib0 + i].qh[l0 +  1] & hm2) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx + 16] * (q4_10 + (((data_a[ib0 + i].qh[l0 + 16] & hm2) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx + 17] * (q4_11 + (((data_a[ib0 + i].qh[l0 + 17] & hm2) != 0) ? 16 : 0))
        );
        const FLOAT_TYPE sw = FLOAT_TYPE(
            data_b[p.b_offset + y2_idx + 32] * (q4_12 + (((data_a[ib0 + i].qh[l0     ] & (hm2 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx + 33] * (q4_13 + (((data_a[ib0 + i].qh[l0 +  1] & (hm2 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx + 48] * (q4_14 + (((data_a[ib0 + i].qh[l0 + 16] & (hm2 << 1)) != 0) ? 16 : 0))
          + data_b[p.b_offset + y2_idx + 49] * (q4_15 + (((data_a[ib0 + i].qh[l0 + 17] & (hm2 << 1)) != 0) ? 16 : 0))
        );
        const FLOAT_TYPE smin = FLOAT_TYPE(
            (data_b[p.b_offset + y1_idx] + data_b[p.b_offset + y1_idx + 1] + data_b[p.b_offset + y1_idx + 16] + data_b[p.b_offset + y1_idx + 17]) * sc2 + (data_b[p.b_offset + y1_idx + 32] + data_b[p.b_offset + y1_idx + 33] + data_b[p.b_offset + y1_idx + 48] + data_b[p.b_offset + y1_idx + 49]) * sc3
          + (data_b[p.b_offset + y2_idx] + data_b[p.b_offset + y2_idx + 1] + data_b[p.b_offset + y2_idx + 16] + data_b[p.b_offset + y2_idx + 17]) * sc6 + (data_b[p.b_offset + y2_idx + 32] + data_b[p.b_offset + y2_idx + 33] + data_b[p.b_offset + y2_idx + 48] + data_b[p.b_offset + y2_idx + 49]) * sc7
        );
        tmp[16 * ix + tid] += FLOAT_TYPE(dall * (sx * sc0 + sy * sc1 + sz * sc4 + sw * sc5) - dmin * smin);
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""
mul_mat_vec_q6_K_body = """
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

shared FLOAT_TYPE tmp[32];

void main() {
    const uint block_size = gl_WorkGroupSize.x;
    const uint row = gl_WorkGroupID.x;

    const uint num_blocks_per_row = p.ncols / QUANT_K;
    const uint ib0 = row*num_blocks_per_row;

    const uint tid = gl_LocalInvocationID.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const uint ix  = gl_LocalInvocationID.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const uint step = 16/K_QUANTS_PER_ITERATION;            // 16 or 8

    const uint v_im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const uint v_in = tid - step*v_im;                      // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const uint l0 = v_in;                                   // 0...15
    const uint is = 0;
#else
    const uint l0 = 4 * v_in;                               // 0, 4, 8, ..., 28
    const uint is = v_in / 4;
#endif

    const uint ql_offset = 64*v_im + l0;
    const uint qh_offset = 32*v_im + l0;
    const uint s_offset  =  8*v_im + is;
    const uint y_offset = 128*v_im + l0;

    tmp[16 * ix + tid] = FLOAT_TYPE(0.0); // partial sum for thread in warp

    [[unroll]] for (uint i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        const uint y_idx   = i * QUANT_K + y_offset;

        const FLOAT_TYPE d = FLOAT_TYPE(data_a[ib0 + i].d);

#if K_QUANTS_PER_ITERATION == 1
        FLOAT_TYPE sum = FLOAT_TYPE(data_b[p.b_offset + y_idx +  0]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 0]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset +  0] & 0xF) | ((data_a[ib0 + i].qh[qh_offset +  0] & 0x03) << 4)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 16]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 1]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 16] & 0xF) | ((data_a[ib0 + i].qh[qh_offset + 16] & 0x03) << 4)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 32]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 2]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 32] & 0xF) | ((data_a[ib0 + i].qh[qh_offset +  0] & 0x0c) << 2)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 48]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 3]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 48] & 0xF) | ((data_a[ib0 + i].qh[qh_offset + 16] & 0x0c) << 2)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 64]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 4]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset +  0]  >> 4) | ((data_a[ib0 + i].qh[qh_offset +  0] & 0x30) >> 0)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 80]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 5]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 16]  >> 4) | ((data_a[ib0 + i].qh[qh_offset + 16] & 0x30) >> 0)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx + 96]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 6]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 32]  >> 4) | ((data_a[ib0 + i].qh[qh_offset +  0] & 0xc0) >> 2)) - 32)
                       + FLOAT_TYPE(data_b[p.b_offset + y_idx +112]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 7]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + 48]  >> 4) | ((data_a[ib0 + i].qh[qh_offset + 16] & 0xc0) >> 2)) - 32);
        tmp[16 * ix + tid] += sum;
#else
        FLOAT_TYPE sum = FLOAT_TYPE(0.0);
        [[unroll]] for (int l = 0; l < 4; ++l) {
            sum += FLOAT_TYPE(data_b[p.b_offset + y_idx + l+ 0]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 0]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + l+ 0] & 0xF) | (((data_a[ib0 + i].qh[qh_offset + l] >> 0) & 3) << 4)) - 32)
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l+32]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 2]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + l+32] & 0xF) | (((data_a[ib0 + i].qh[qh_offset + l] >> 2) & 3) << 4)) - 32)
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l+64]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 4]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + l+ 0]  >> 4) | (((data_a[ib0 + i].qh[qh_offset + l] >> 4) & 3) << 4)) - 32)
                 + FLOAT_TYPE(data_b[p.b_offset + y_idx + l+96]) * FLOAT_TYPE(data_a[ib0 + i].scales[s_offset + 6]) * d * FLOAT_TYPE(int8_t((data_a[ib0 + i].ql[ql_offset + l+32]  >> 4) | (((data_a[ib0 + i].qh[qh_offset + l] >> 6) & 3) << 4)) - 32);
        }
        tmp[16 * ix + tid] += sum;
#endif
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
       }
        barrier();
    }
    if (tid == 0) {
        dst[p.d_offset + row] = D_TYPE(tmp[0]);
    }
}
"""

mul_mat_p021_src = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#define BLOCK_SIZE 32
#define FLOAT_TYPE float

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

layout (push_constant) uniform parameter
{
    uint ncols_x;
    uint nrows_x;
    uint nchannels_x;
    uint nchannels_y;
    uint b_offset;
    uint d_offset;
} p;

shared FLOAT_TYPE tmp[BLOCK_SIZE];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint row_x = gl_GlobalInvocationID.y;
    const uint channel = gl_GlobalInvocationID.z;
    const uint channel_x = channel / (p.nchannels_y / p.nchannels_x);

    const uint nrows_y = p.ncols_x;
    const uint nrows_dst = p.nrows_x;
    const uint row_dst = row_x;

    tmp[tid] = FLOAT_TYPE(0.0f);

    for (uint col_x0 = 0; col_x0 < p.ncols_x; col_x0 += BLOCK_SIZE) {
        const uint col_x = col_x0 + tid;

        if (col_x >= p.ncols_x) {
            break;
        }

        // x is transposed and permuted
        const uint ix = row_x*p.nchannels_x*p.ncols_x + channel_x*p.ncols_x + col_x;
        const FLOAT_TYPE xi = FLOAT_TYPE(data_a[ix]);

        const uint row_y = col_x;

        // y is not transposed but permuted
        const uint iy = channel*nrows_y + row_y;

        tmp[tid] += xi * FLOAT_TYPE(data_b[iy]);
    }

    // dst is not transposed and not permuted
    const uint idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        dst[idst] = tmp[0];
    }
}
"""


mul_mat_nc_src = """#version 450

#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require

#define BLOCK_SIZE 32
#define FLOAT_TYPE float

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst[];};

layout (push_constant) uniform parameter
{
    uint ncols_x;
    uint nrows_x;
    uint row_stride_x;
    uint channel_stride_x;
    uint channel_x_divisor;
    uint b_offset;
    uint d_offset;
} p;

shared FLOAT_TYPE tmp[BLOCK_SIZE];

void main() {
    const uint tid       = gl_LocalInvocationID.x;
    const uint row_x     = gl_GlobalInvocationID.y;
    const uint channel   = gl_GlobalInvocationID.z;
    const uint channel_x = channel / p.channel_x_divisor;

    const uint nrows_y   = p.ncols_x;
    const uint nrows_dst = p.nrows_x;
    const uint row_dst   = row_x;

    const uint idst = channel*nrows_dst + row_dst;

    tmp[tid] = 0.0f;

    for (uint col_x0 = 0; col_x0 < p.ncols_x; col_x0 += BLOCK_SIZE) {
        const uint col_x = col_x0 + tid;

        if (col_x >= p.ncols_x) {
            break;
        }

        const uint row_y = col_x;

        const uint ix = channel_x*p.channel_stride_x + row_x*p.row_stride_x + col_x;
        const uint iy = channel*nrows_y + row_y;

        const FLOAT_TYPE xi = FLOAT_TYPE(data_a[ix]);

        tmp[tid] += xi * FLOAT_TYPE(data_b[iy]);
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        barrier();
    }

    if (tid == 0) {
        dst[idst] = tmp[0];
    }
}
"""

generic_head = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    float param1;
    float param2;
} p;
"""

generic_unary_op_head = """#version 450

#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint d_offset;
    float param1; float param2;
} p;"""

generic_unary_op_funcs = """
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

uint src0_idx(uint idx) {
    const uint i03 = idx / (p.ne02*p.ne01*p.ne00);
    const uint i03_offset = i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = (idx - i03_offset) / (p.ne01*p.ne00);
    const uint i02_offset = i02*p.ne01*p.ne00;
    const uint i01 = (idx - i03_offset - i02_offset) / p.ne00;
    const uint i00 = idx - i03_offset - i02_offset - i01*p.ne00;
    return i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
}

uint dst_idx(uint idx) {
    const uint i13 = idx / (p.ne12*p.ne11*p.ne10);
    const uint i13_offset = i13 * p.ne12*p.ne11*p.ne10;
    const uint i12 = (idx - i13_offset) / (p.ne11*p.ne10);
    const uint i12_offset = i12*p.ne11*p.ne10;
    const uint i11 = (idx - i13_offset - i12_offset) / p.ne10;
    const uint i10 = idx - i13_offset - i12_offset - i11*p.ne10;
    return i13*p.nb13 + i12*p.nb12 + i11*p.nb11 + i10*p.nb10;
}"""

generic_unary_op_main = """
void main() {
    if (gl_GlobalInvocationID.x >= p.ne) {
        return;
    }
"""

generic_unary_op_combined = f"{generic_unary_op_head}\n{generic_unary_op_funcs}\n{generic_unary_op_main}"

generic_binary_op_head = """#version 450

#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint ne20; uint ne21; uint ne22; uint ne23; uint nb20; uint nb21; uint nb22; uint nb23;
    uint d_offset;
    float param1; float param2;
} p;"""

generic_binary_op_funcs = """
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

uint src0_idx(uint idx) {
    const uint i03 = idx / (p.ne02*p.ne01*p.ne00);
    const uint i03_offset = i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = (idx - i03_offset) / (p.ne01*p.ne00);
    const uint i02_offset = i02*p.ne01*p.ne00;
    const uint i01 = (idx - i03_offset - i02_offset) / p.ne00;
    const uint i00 = idx - i03_offset - i02_offset - i01*p.ne00;
    return i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
}

uint src1_idx(uint idx) {
    const uint i03 = idx / (p.ne02*p.ne01*p.ne00);
    const uint i03_offset = i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = (idx - i03_offset) / (p.ne01*p.ne00);
    const uint i02_offset = i02*p.ne01*p.ne00;
    const uint i01 = (idx - i03_offset - i02_offset) / p.ne00;
    const uint i00 = idx - i03_offset - i02_offset - i01*p.ne00;

    return (i03 % p.ne13)*p.nb13 + (i02 % p.ne12)*p.nb12 + (i01 % p.ne11)*p.nb11 + (i00 % p.ne10)*p.nb10;
}

uint dst_idx(uint idx) {
    const uint i23 = idx / (p.ne22*p.ne21*p.ne20);
    const uint i23_offset = i23 * p.ne22*p.ne21*p.ne20;
    const uint i22 = (idx - i23_offset) / (p.ne21*p.ne20);
    const uint i22_offset = i22*p.ne21*p.ne20;
    const uint i21 = (idx - i23_offset - i22_offset) / p.ne20;
    const uint i20 = idx - i23_offset - i22_offset - i21*p.ne20;
    return i23*p.nb23 + i22*p.nb22 + i21*p.nb21 + i20*p.nb20;
}"""

generic_binary_op_main = """
void main() {
    if (gl_GlobalInvocationID.x >= p.ne) {
        return;
    }
"""

generic_binary_op_combined = f"{generic_binary_op_head}\n{generic_binary_op_funcs}\n{generic_binary_op_main}"

# MUL F32
mul_body = """
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(FLOAT_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]) * FLOAT_TYPE(data_b[src1_idx(gl_GlobalInvocationID.x)]));
}
"""

# ADD
add_body = """
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(FLOAT_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]) + FLOAT_TYPE(data_b[src1_idx(gl_GlobalInvocationID.x)]));
}
"""

# SCALE
scale_body = """
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(FLOAT_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]) + FLOAT_TYPE(p.param1));
}
"""

# SQR
sqr_body = """
    const FLOAT_TYPE val = FLOAT_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]);
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(val * val);
}
"""

# CLAMP
clamp_body = """
    const FLOAT_TYPE val = FLOAT_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]);
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(val < p.param1 ? p.param1 : (val > p.param2 ? p.param2 : val));
}
"""

# CPY
cpy_end = """
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = D_TYPE(data_a[src0_idx(gl_GlobalInvocationID.x)]);
}
"""
# Causes an optimization error otherwise
cpy_f16_f16_end = """
    data_d[p.d_offset + dst_idx(gl_GlobalInvocationID.x)] = data_a[src0_idx(gl_GlobalInvocationID.x)];
}
"""

# GET_ROWS
get_rows_float_body = """
void main() {
    const uint i00 = gl_GlobalInvocationID.x;
    const uint i10 = gl_GlobalInvocationID.y;
    const uint i11 = (gl_GlobalInvocationID.z)/p.ne12;
    const uint i12 = (gl_GlobalInvocationID.z)%p.ne12;

    if (i00 >= p.ne00) {
        return;
    }

    const uint i01 = data_b[i10*p.nb10 + i11*p.nb11 + i12*p.nb12];

    const uint a_offset = i01*p.nb01 + i11*p.nb02 + i12*p.nb03;
    const uint d_offset = i10*p.nb21 + i11*p.nb22 + i12*p.nb23;

#ifndef OPTIMIZATION_ERROR_WORKAROUND
    data_d[d_offset + i00] = D_TYPE(data_a[a_offset + i00]);
#else
    data_d[d_offset + i00] = data_a[a_offset + i00];
#endif
}
"""

get_rows_body = """
void main() {
    const uint i00 = (gl_GlobalInvocationID.x)*2;
    const uint i10 = gl_GlobalInvocationID.y;
    const uint i11 = (gl_GlobalInvocationID.z)/p.ne12;
    const uint i12 = (gl_GlobalInvocationID.z)%p.ne12;

    if (i00 >= p.ne00) {
        return;
    }

    const uint i01 = data_b[i10*p.nb10 + i11*p.nb11 + i12*p.nb12];

    const uint a_offset = i01*p.nb01 + i11*p.nb02 + i12*p.nb03;
    const uint d_offset = i10*p.nb21 + i11*p.nb22 + i12*p.nb23;

    const uint ib = a_offset + i00/QUANT_K; // block index
    const uint iqs = (i00%QUANT_K)/QUANT_R; // quant index
    const uint iybs = i00 - i00%QUANT_K; // dst block start index
    const uint y_offset = QUANT_R == 1 ? 1 : QUANT_K/2;

    DEQUANT_FUNC

    data_d[d_offset + iybs + iqs           ] = D_TYPE(v.x);
    data_d[d_offset + iybs + iqs + y_offset] = D_TYPE(v.y);
}
"""

# UNARY
gelu_body = """
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

void main() {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const uint i = gl_GlobalInvocationID.x;

    if (i >= p.KX) {
        return;
    }

    const float xi = float(data_a[i]);
    const float val = SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi);
    data_d[i] = D_TYPE(0.5f*xi*(2.0f - 2.0f / (exp(2 * val) + 1)));
}
"""

silu_body = """
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

void main() {
    const uint i = gl_GlobalInvocationID.x;

    if (i >= p.KX) {
        return;
    }

    const float xi = float(data_a[i]);
    data_d[i] = D_TYPE(xi / (1.0f + exp(-xi)));
}
"""

relu_body = """
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

void main() {
    const uint i = gl_GlobalInvocationID.x;

    if (i >= p.KX) {
        return;
    }

    data_d[i] = max(float(data_a[i]), 0);
}
"""

# DIAG_MASK_INF
diag_mask_inf_head = """#version 450

#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint ncols;
    uint rows_per_channel;
    uint n_past;
} p;
"""
diag_mask_inf_body = """
#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

void main() {
    const uint col = gl_GlobalInvocationID.y;
    const uint row = gl_GlobalInvocationID.x;

    if (col >= p.ncols) {
        return;
    }

    const uint i = row*p.ncols + col;
    data_d[i] = D_TYPE(data_a[i] - float(uint(col > p.n_past + row % p.rows_per_channel) * 0xFFFFFFFF));
}
"""

# NORMS
norm_body = """
#extension GL_EXT_control_flow_attributes : enable
#define BLOCK_SIZE 512

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

shared vec2 sum[BLOCK_SIZE];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    const float eps = 1e-5f;

    sum[tid] = vec2(0.0f, 0.0f);

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const float xi = float(data_a[row*p.KX + col]);
        sum[tid].x += xi;
        sum[tid].y += xi * xi;
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        barrier();
    }

    const float mean = sum[0].x / p.KX;
    const float var = sum[0].y / p.KX - mean * mean;
    const float inv_std = inversesqrt(var + 1e-5f);

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[row*p.KX + col] = D_TYPE((float(data_a[row*p.KX + col]) - mean) * inv_std);
    }
}
"""

rms_norm_body = """
#extension GL_EXT_control_flow_attributes : enable
#define BLOCK_SIZE 512

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

shared FLOAT_TYPE sum[BLOCK_SIZE];

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;

    sum[tid] = FLOAT_TYPE(0.0f); // partial sum for thread in warp

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const FLOAT_TYPE xi = FLOAT_TYPE(data_a[row*p.KX + col]);
        sum[tid] += xi * xi;
    }

    // sum up partial sums and write back result
    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        barrier();
    }

    const FLOAT_TYPE mean = sum[0] / FLOAT_TYPE(p.KX);
    const FLOAT_TYPE scale = inversesqrt(mean + FLOAT_TYPE(p.param1));

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[row*p.KX + col] = D_TYPE(scale * FLOAT_TYPE(data_a[row*p.KX + col]));
    }
}
"""

# SOFT_MAX
soft_max_head = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    uint KZ;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint n_head_log2;
} p;
"""

soft_max_body = """
#extension GL_EXT_control_flow_attributes : enable
#define BLOCK_SIZE 512

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) readonly buffer Y {B_TYPE data_b[];};
layout (binding = 2) readonly buffer Z {C_TYPE data_c[];};
layout (binding = 3) buffer D {D_TYPE data_d[];};

shared FLOAT_TYPE vals[BLOCK_SIZE];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint rowx = gl_WorkGroupID.x;
    const uint rowy = rowx % p.KY;

    float slope = 0.0f;

    // ALiBi
    if (p.max_bias > 0.0f) {
        const uint h = rowx/p.KY; // head index

        const float base = h < p.n_head_log2 ? p.m0 : p.m1;
        const uint   exp  = h < p.n_head_log2 ? h + 1 : 2*(h - p.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // Find max
    vals[tid] = uintBitsToFloat(0xFF800000);

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        vals[tid] = max(vals[tid], FLOAT_TYPE(data_a[rowx * p.KX + col]) * p.scale + (p.KY > 0 ? FLOAT_TYPE(data_b[rowy * p.KX + col]) : FLOAT_TYPE(0.0f)) + (p.KZ > 0 ? slope * data_c[col] : 0.0f));
    }

    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            vals[tid] = max(vals[tid], vals[tid + s]);
        }
        barrier();
    }

    const FLOAT_TYPE max_val = vals[0];
    barrier();

    // Sum up values
    vals[tid] = FLOAT_TYPE(0.0f);

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        const uint i = rowx * p.KX + col;
        const FLOAT_TYPE val = exp(FLOAT_TYPE(data_a[i]) * p.scale + (p.KY > 0 ? FLOAT_TYPE(data_b[rowy * p.KX + col]) : FLOAT_TYPE(0.0f)) - max_val);
        vals[tid] += val;
        data_d[i] = D_TYPE(val);
    }

    barrier();
    [[unroll]] for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            vals[tid] += vals[tid + s];
        }
        barrier();
    }

    const D_TYPE divisor = D_TYPE(vals[0]);

    [[unroll]] for (uint col = tid; col < p.KX; col += BLOCK_SIZE) {
        data_d[rowx*p.KX + col] /= divisor;
    }
}
"""

# ROPE
rope_src = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 1, local_size_y = 256, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) readonly buffer Y {int data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (push_constant) uniform parameter {
    uint ncols;
    float freq_scale;
    uint p_delta_rows;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[4];
} p;

float rope_yarn_ramp(const float low, const float high, const uint i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

void rope_yarn(const float theta_extrap, const uint i0, out float cos_theta, out float sin_theta) {
    float mscale = p.attn_factor;
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    if (p.ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(p.corr_dims[0], p.corr_dims[1], i0) * p.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

void main() {
    const uint col = gl_GlobalInvocationID.y * 2;
    const uint row = gl_GlobalInvocationID.x;

    if (col >= p.ncols) {
        return;
    }

    const uint i = row*p.ncols + col;
    const uint i2 = row/p.p_delta_rows;

    const int pos = data_b[i2];
    const float theta_base = pos * pow(p.freq_base, -float(col)/p.ncols);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, col, cos_theta, sin_theta);

    const float x0 = float(data_a[i + 0]);
    const float x1 = float(data_a[i + 1]);

    data_d[i + 0] = D_TYPE(x0*cos_theta - x1*sin_theta);
    data_d[i + 1] = D_TYPE(x0*sin_theta + x1*cos_theta);
}
"""

rope_neox_src = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 1, local_size_y = 256, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) readonly buffer Y {int data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (push_constant) uniform parameter {
    uint ncols;
    uint ndims;
    float freq_scale;
    uint p_delta_rows;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[4];
    float theta_scale;
    float inv_ndims;
} p;

float rope_yarn_ramp(const float low, const float high, const uint i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

void rope_yarn(const float theta_extrap, const uint i0, out float cos_theta, out float sin_theta) {
    float mscale = p.attn_factor;
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = p.freq_scale * theta_extrap;
    float theta = theta_interp;
    if (p.ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(p.corr_dims[0], p.corr_dims[1], i0) * p.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / p.freq_scale);
    }
    cos_theta = cos(theta) * mscale;
    sin_theta = sin(theta) * mscale;
}

void main() {
    const uint col = gl_GlobalInvocationID.y * 2;
    const uint row = gl_GlobalInvocationID.x;

    if (col >= p.ncols) {
        return;
    }

    const uint ib = col / p.ndims;
    const uint ic = col % p.ndims;

    if (ib > 0) {
        const uint i = row*p.ncols + ib*p.ndims + ic;

        data_d[i + 0] = data_a[i + 0];
        data_d[i + 1] = data_a[i + 1];

        return;
    }

    const uint i  = row*p.ncols + ib*p.ndims + ic/2;
    const uint i2 = row/p.p_delta_rows;

    const float cur_rot = p.inv_ndims * ic - ib;

    const int pos = data_b[i2];
    const float theta_base = pos*p.freq_scale*pow(p.theta_scale, col/2.0f);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, uint(cur_rot), cos_theta, sin_theta);

    const float x0 = float(data_a[i + 0]);
    const float x1 = float(data_a[i + p.ndims/2]);

    data_d[i + 0]        = D_TYPE(x0*cos_theta - x1*sin_theta);
    data_d[i + p.ndims/2] = D_TYPE(x0*sin_theta + x1*cos_theta);
}
"""

argsort_src = """
#version 450

#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1)          buffer D {int data_d[];};

layout (push_constant) uniform parameter {
    uint ncols;
    bool ascending;
} p;

void swap(uint idx0, uint idx1) {
    int tmp = data_d[idx0];
    data_d[idx0] = data_d[idx1];
    data_d[idx1] = tmp;
}

void main() {
    // bitonic sort
    const int col = int(gl_LocalInvocationID.x);
    const uint row = gl_WorkGroupID.y;

    if (col >= p.ncols) {
        return;
    }

    const uint a_idx = row * p.ncols;
    const uint d_idx = row * p.ncols;

    // initialize indices
    if (col < p.ncols) {
        data_d[col] = col;
    }
    barrier();

    for (uint k = 2; k <= p.ncols; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            const uint ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (p.ascending ? data_a[a_idx + data_d[d_idx + col]] > data_a[a_idx + data_d[d_idx + ixj]] : data_a[a_idx + data_d[d_idx + col]] < data_a[a_idx + data_d[d_idx + ixj]]) {
                        swap(d_idx + col, d_idx + ixj);
                    }
                } else {
                    if (p.ascending ? data_a[a_idx + data_d[d_idx + col]] < data_a[a_idx + data_d[d_idx + ixj]] : data_a[a_idx + data_d[d_idx + col]] > data_a[a_idx + data_d[d_idx + ixj]]) {
                        swap(d_idx + col, d_idx + ixj);
                    }
                }
            }
            barrier();
        }
    }
}
"""

GLSLC = "glslc"

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

K_QUANTS_PER_ITERATION = 2

ASYNCIO_CONCURRENCY = 64

output_dir = gettempdir()

lock = asyncio.Lock()
shader_fnames = []


async def string_to_spv(name, code, defines, fp16=True):
    f = NamedTemporaryFile(mode="w", delete=False)
    f.write(code)
    f.flush()

    name = f"{name}{'_fp32' if not fp16 else ''}"
    fname = os.path.join(output_dir, f"{name}.comp")

    cmd = [GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", f.name, "-o", fname]

    cmd.extend([f"-D{key}={value}" for key, value in defines.items()])

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    stdout, stderr = await proc.communicate()

    stdout = stdout.decode()
    error = stderr.decode()

    if proc.returncode:
        # Generate preprocessed code
        cmd = [GLSLC, "-E", f.name]
        cmd.extend([f"-D{key}={value}" for key, value in defines.items()])

        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()

        logger.info(" ".join(cmd))

        if proc.returncode:
            raise RuntimeError(f"{name=} {f.name=} {stdout=} {stderr=}")

        preprocessed_code = stdout.decode()

        cmd.extend([f"-D{key}={value}" for key, value in defines.items()])
        code_with_lines = "\n".join([f"{i + 1}: {line}" for i, line in enumerate(preprocessed_code.splitlines())])
        logger.error(f"cannot compile {name}\n\n{code_with_lines}\n\n{error}")
        f.close()
        os.remove(f.name)
        sys.exit(proc.returncode)

    f.close()
    os.remove(f.name)

    async with lock:
        shader_fnames.append((name, fname))


async def main():
    logger.info("ggml_vulkan: Generating and compiling shaders to SPIR-V")

    tasks = []

    stream = []

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

        stream.clear()
        stream.extend((mulmat_head, shader_float_type, mulmat_body1, mulmat_load_scalar, mulmat_body2))
        tasks.append(string_to_spv("matmul_f32", "".join(stream), {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_f32_aligned", "".join(stream), {"LOAD_VEC_A": load_vec, "LOAD_VEC_B": load_vec, "A_TYPE": vec_type, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        tasks.append(string_to_spv("matmul_f16", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float16_t", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_f16_aligned", "".join(stream), {"LOAD_VEC_A": load_vec, "LOAD_VEC_B": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type_f16, "D_TYPE": "float"}, fp16))

        tasks.append(string_to_spv("matmul_f16_f32", "".join(stream), {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_f16_f32_aligned", "".join(stream), {"LOAD_VEC_A": load_vec, "LOAD_VEC_B": load_vec, "A_TYPE": vec_type_f16, "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q4_0_defines, mulmat_body1, mulmat_load_q4_0, mulmat_body2))
        tasks.append(string_to_spv("matmul_q4_0_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q4_0", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q4_0_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q4_0", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q4_1_defines, mulmat_body1, mulmat_load_q4_1, mulmat_body2))
        tasks.append(string_to_spv("matmul_q4_1_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q4_1", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q4_1_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q4_1", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q5_0_defines, mulmat_body1, mulmat_load_q5_0, mulmat_body2))
        tasks.append(string_to_spv("matmul_q5_0_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q5_0", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q5_0_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q5_0", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q5_1_defines, mulmat_body1, mulmat_load_q5_1, mulmat_body2))
        tasks.append(string_to_spv("matmul_q5_1_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q5_1", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q5_1_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q5_1", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q8_0_defines, mulmat_body1, mulmat_load_q8_0, mulmat_body2))
        tasks.append(string_to_spv("matmul_q8_0_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q8_0", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q8_0_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q8_0", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q2_K_defines, mulmat_body1, mulmat_load_q2_K, mulmat_body2))
        tasks.append(string_to_spv("matmul_q2_k_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q2_K", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q2_k_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q2_K", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q3_K_defines, mulmat_body1, mulmat_load_q3_K, mulmat_body2))
        tasks.append(string_to_spv("matmul_q3_k_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q3_K", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q3_k_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q3_K", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q4_K_defines, mulmat_body1, mulmat_load_q4_K, mulmat_body2))
        tasks.append(string_to_spv("matmul_q4_k_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q4_K", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q4_k_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q4_K", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q5_K_defines, mulmat_body1, mulmat_load_q5_K, mulmat_body2))
        tasks.append(string_to_spv("matmul_q5_k_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q5_K", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q5_k_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q5_K", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

        stream.clear()
        stream.extend((mulmat_head, shader_int8_ext, shader_float_type, shader_q6_K_defines, mulmat_body1, mulmat_load_q6_K, mulmat_body2))
        tasks.append(string_to_spv("matmul_q6_k_f32", "".join(stream), {"LOAD_VEC_A": 2, "A_TYPE": "block_q6_K", "B_TYPE": "float", "D_TYPE": "float"}, fp16))
        tasks.append(string_to_spv("matmul_q6_k_f32_aligned", "".join(stream), {"LOAD_VEC_A": 2, "LOAD_VEC_B": load_vec, "A_TYPE": "block_q6_K", "B_TYPE": vec_type, "D_TYPE": "float"}, fp16))

    # Shaders where precision is needed, so no fp16 version

    # mul mat vec
    for i in range(0, VK_NUM_TYPES):
        stream.clear()
        stream.extend((mul_mat_vec_head, shader_int8_ext, shader_f32))

        if i == GGML_TYPE_F16:
            stream.extend((shader_f16_defines, shader_float_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q4_0:
            stream.extend((shader_q4_0_defines, shader_q4_0_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q4_1:
            stream.extend((shader_q4_1_defines, shader_q4_1_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q5_0:
            stream.extend((shader_q5_0_defines, shader_q5_0_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q5_1:
            stream.extend((shader_q5_1_defines, shader_q5_1_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q8_0:
            stream.extend((shader_q8_0_defines, shader_q8_0_dequant_func, mul_mat_vec_body))
        elif i == GGML_TYPE_Q2_K:
            stream.extend((shader_q2_K_defines, mul_mat_vec_q2_K_body))
        elif i == GGML_TYPE_Q3_K:
            stream.extend((shader_q3_K_defines, mul_mat_vec_q3_K_body))
        elif i == GGML_TYPE_Q4_K:
            stream.extend((shader_q4_K_defines, mul_mat_vec_q4_K_body))
        elif i == GGML_TYPE_Q5_K:
            stream.extend((shader_q5_K_defines, mul_mat_vec_q5_K_body))
        elif i == GGML_TYPE_Q6_K:
            stream.extend((shader_q6_K_defines, mul_mat_vec_q6_K_body))
        else:
            continue

        tasks.append(string_to_spv(f"mul_mat_vec_{type_names[i]}_f32", "".join(stream), {"B_TYPE": "float", "D_TYPE": "float", "K_QUANTS_PER_ITERATION": K_QUANTS_PER_ITERATION}))

    # Dequant shaders
    for i in range(0, VK_NUM_TYPES):
        stream.clear()

        stream.extend((dequant_head, shader_int8_ext, shader_f32))

        if i == GGML_TYPE_F32:
            stream.append(dequant_f32_body)
        elif i == GGML_TYPE_Q4_0:
            stream.extend((shader_q4_0_defines, dequant_q4_0_body))
        elif i == GGML_TYPE_Q4_1:
            stream.extend((shader_q4_1_defines, dequant_q4_1_body))
        elif i == GGML_TYPE_Q5_0:
            stream.extend((shader_q5_0_defines, dequant_q5_0_body))
        elif i == GGML_TYPE_Q5_1:
            stream.extend((shader_q5_1_defines, dequant_q5_1_body))
        elif i == GGML_TYPE_Q8_0:
            stream.extend((shader_q8_0_defines, dequant_q8_0_body))
        elif i == GGML_TYPE_Q2_K:
            stream.extend((shader_q2_K_defines, dequant_q2_K_body))
        elif i == GGML_TYPE_Q3_K:
            stream.extend((shader_q3_K_defines, dequant_q3_K_body))
        elif i == GGML_TYPE_Q4_K:
            stream.extend((shader_q4_K_defines, dequant_q4_K_body))
        elif i == GGML_TYPE_Q5_K:
            stream.extend((shader_q5_K_defines, dequant_q5_K_body))
        elif i == GGML_TYPE_Q6_K:
            stream.extend((shader_q6_K_defines, dequant_q6_K_body))
        else:
            continue

        tasks.append(string_to_spv(f"dequant_{type_names[i]}", "".join(stream), {"D_TYPE": "float16_t"}))

    # get_rows
    for i in range(0, VK_NUM_TYPES):
        stream.clear()
        stream.extend((generic_binary_op_head, shader_int8_ext, shader_f32))
        optimization_workaround = False

        if i == GGML_TYPE_F32:
            stream.extend((shader_f32_defines, generic_binary_op_funcs, get_rows_float_body))
        elif i == GGML_TYPE_F16:
            stream.extend((shader_f16_defines, generic_binary_op_funcs, get_rows_float_body))
            optimization_workaround = True
        elif i == GGML_TYPE_Q4_0:
            stream.extend((shader_q4_0_defines, shader_q4_0_dequant_func, generic_binary_op_funcs, get_rows_body))
        elif i == GGML_TYPE_Q4_1:
            stream.extend((shader_q4_1_defines, shader_q4_1_dequant_func, generic_binary_op_funcs, get_rows_body))
        elif i == GGML_TYPE_Q5_0:
            stream.extend((shader_q5_0_defines, shader_q5_0_dequant_func, generic_binary_op_funcs, get_rows_body))
        elif i == GGML_TYPE_Q5_1:
            stream.extend((shader_q5_1_defines, shader_q5_1_dequant_func, generic_binary_op_funcs, get_rows_body))
        elif i == GGML_TYPE_Q8_0:
            stream.extend((shader_q8_0_defines, shader_q8_0_dequant_func, generic_binary_op_funcs, get_rows_body))
        else:
            continue

        if optimization_workaround:
            tasks.append(string_to_spv(f"get_rows_{type_names[i]}", "".join(stream), {"B_TYPE": "int", "D_TYPE": "float16_t", "OPTIMIZATION_ERROR_WORKAROUND": "1"}))
        else:
            tasks.append(string_to_spv(f"get_rows_{type_names[i]}", "".join(stream), {"B_TYPE": "int", "D_TYPE": "float16_t"}))
        tasks.append(string_to_spv(f"get_rows_{type_names[i]}_f32", "".join(stream), {"B_TYPE": "int", "D_TYPE": "float"}))

    tasks.append(string_to_spv("mul_mat_vec_p021_f16_f32", mul_mat_p021_src, {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("mul_mat_vec_nc_f16_f32", mul_mat_nc_src, {"A_TYPE": "float16_t", "B_TYPE": "float", "D_TYPE": "float"}))

    # Norms
    tasks.append(string_to_spv("norm_f32", f"{generic_head}\n{shader_f32}\n{norm_body}", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rms_norm_f32", f"{generic_head}\n{shader_f32}\n{rms_norm_body}", {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("cpy_f32_f32", f"{generic_unary_op_combined}\n{cpy_end}", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("cpy_f32_f16", f"{generic_unary_op_combined}\n{cpy_end}", {"A_TYPE": "float", "D_TYPE": "float16_t"}))
    tasks.append(string_to_spv("cpy_f16_f16", f"{generic_unary_op_combined}\n{cpy_f16_f16_end}", {"A_TYPE": "float16_t", "D_TYPE": "float16_t"}))

    tasks.append(string_to_spv("add_f32", f"{generic_binary_op_combined}\n{add_body}", {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("split_k_reduce", mulmat_split_k_reduce_src, {}))
    tasks.append(string_to_spv("mul_f32", f"{generic_binary_op_combined}\n{mul_body}", {"A_TYPE": "float", "B_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("scale_f32", f"{generic_unary_op_combined}\n{scale_body}", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("sqr_f32", f"{generic_unary_op_combined}\n{sqr_body}", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("clamp_f32", f"{generic_unary_op_combined}\n{clamp_body}", {"A_TYPE": "float", "D_TYPE": "float", "FLOAT_TYPE": "float"}))

    tasks.append(string_to_spv("gelu_f32", f"{generic_head}\n{shader_f32}\n{gelu_body}", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("silu_f32", f"{generic_head}\n{shader_f32}\n{silu_body}", {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("relu_f32", f"{generic_head}\n{shader_f32}\n{relu_body}", {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("diag_mask_inf_f32", f"{diag_mask_inf_head}\n{shader_f32}\n{diag_mask_inf_body}", {"A_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("soft_max_f32", f"{soft_max_head}\n{shader_f32}\n{soft_max_body}", {"A_TYPE": "float", "B_TYPE": "float", "C_TYPE": "float", "D_TYPE": "float"}))

    tasks.append(string_to_spv("rope_f32", rope_src, {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rope_f16", rope_src, {"A_TYPE": "float16_t", "D_TYPE": "float16_t"}))

    tasks.append(string_to_spv("rope_neox_f32", rope_neox_src, {"A_TYPE": "float", "D_TYPE": "float"}))
    tasks.append(string_to_spv("rope_neox_f16", rope_neox_src, {"A_TYPE": "float16_t", "D_TYPE": "float16_t"}))

    tasks.append(string_to_spv("argsort_f32", argsort_src, {"A_TYPE": "float"}))

    # Helper to decorate tasks with semaphore acquisition.
    async def withSemaphore(sem, task):
        async with sem:
            return await task

    # Run tasks concurrently guarded by a concurrency limit.
    sem = asyncio.Semaphore(ASYNCIO_CONCURRENCY)
    await asyncio.gather(*(withSemaphore(sem, task) for task in tasks))

    with open("ggml-vulkan-shaders.hpp", "w") as f:
        f.write("#include <cstdint>\n\n")
        for name, path in sorted(shader_fnames):

            with open(path, "rb") as spv:
                counter = 0
                newline_counter = 0
                f.write(f"unsigned char {name}_data[] = {{\n")
                for val in spv.read():
                    f.write(f"0x{val:02x},")
                    newline_counter += 1
                    counter += 1
                    if newline_counter >= 12:
                        newline_counter = 0
                        f.write("\n")
            f.write("\n};\n")
            f.write(f"const uint64_t {name}_len = {counter};\n\n")
            os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGML Vulkan Shader Generator")

    parser.add_argument("--glslc", help="Path to glslc")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.glslc:
        GLSLC = args.glslc

    asyncio.run(main())
