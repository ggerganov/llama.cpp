#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char * clblast_dequant = MULTILINE_QUOTE(

struct block_q4_0
{
    float d;
    uchar qs[16];
};

__kernel void dequantize_row_q4_0(__global struct block_q4_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}

struct block_q4_1
{
    float d;
    float m;
    uchar qs[16];
};

__kernel void dequantize_row_q4_1(__global struct block_q4_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;
    const float m = blocks[i].m;

    const uchar vi = blocks[i].qs[l];

    const uint index = i*32 + l*2;
    result[index + 0] = (vi & 0xf) * d + m;
    result[index + 1] = (vi >> 4) * d + m;
}

struct block_q4_2
{
    ushort d;
    uchar qs[8];
};

__kernel void dequantize_row_q4_2(__global struct block_q4_2* blocks, __global float* result) {
    const uint i = get_global_id(0) / 16;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);

    const uchar vi = blocks[i].qs[l];

    const uint index = i*16 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}


struct block_q5_0
{
    float d;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_0(__global struct block_q5_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = (((vi & 0xf) | vh0) - 16)*d;
    result[index + 1] = (((vi >>  4) | vh1) - 16)*d;
}

struct block_q5_1
{
    ushort d;
    ushort m;
    uint qh;
    uchar qs[16];
};

__kernel void dequantize_row_q5_1(__global struct block_q5_1* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = vload_half(0, (__global half*) &blocks[i].d);
    const float m = vload_half(0, (__global half*) &blocks[i].m);

    const uchar vi = blocks[i].qs[l];

    const uint l2 = l * 2;

    const uchar vh0 = ((blocks[i].qh & (1 << (l2 + 0))) >> (l2 + 0)) << 4;
    const uchar vh1 = ((blocks[i].qh & (1 << (l2 + 1))) >> (l2 + 1)) << 4;

    const uint index = i*32 + l2;
    result[index + 0] = ((vi & 0xf) | vh0)*d + m;
    result[index + 1] = ((vi >>  4) | vh1)*d + m;
}

struct block_q8_0
{
    float d;
    uchar qs[32];
};

__kernel void dequantize_row_q8_0(__global struct block_q8_0* blocks, __global float* result) {
    const uint i = get_global_id(0) / 32;
    const uint l = get_local_id(0);

    const float d = blocks[i].d;

    const uint index = i*32 + l;
    result[index] = blocks[i].qs[l] * d;
}

);
