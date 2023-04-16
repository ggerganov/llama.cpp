#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char * clblast_dequant = MULTILINE_QUOTE(

struct __attribute__ ((packed)) block_q4_0
{
    float d;
    uchar qs[16];
};

__kernel void dequantize_row_q4_0(__global struct block_q4_0* blocks, __global float* result) {
    uint i, l;
    i = get_global_id(0) / 32;
    l = get_local_id(0);

    float d = blocks[i].d;

    uchar vi = blocks[i].qs[l];

    uint index = i*32 + l*2;
    result[index + 0] = ((vi & 0xf) - 8)*d;
    result[index + 1] = ((vi >> 4) - 8)*d;
}

struct __attribute__ ((packed)) block_q4_1
{
    float d;
    float m;
    uchar qs[16];
};

__kernel void dequantize_row_q4_1(__global struct block_q4_1* blocks, __global float* result) {
    uint i, l;
    i = get_global_id(0) / 32;
    l = get_local_id(0);

    float d = blocks[i].d;
    float m = blocks[i].m;

    uchar vi = blocks[i].qs[l];

    uint index = i*32 + l*2;
    result[index + 0] = (vi & 0xf) * d + m;
    result[index + 1] = (vi >> 4) * d + m;
}

);