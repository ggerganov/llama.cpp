
#define pair_t u16vec2
#define PAIR_CONST(c) u16vec2(uint16_t(c))
f16vec2 pair_to_f16(const pair_t pair) {
    return f16vec2(uint16BitsToFloat16(pair.x), uint16BitsToFloat16(pair.y));
}

pair_t pack_byte_pair(const uint dword0, const uint dword1, const uint position0, const uint position1) {
    return u16vec2(uint16_t(dword0 >> ((position0 & 3u) * 8u)), uint16_t(dword1 >> ((position1 & 3u) * 8u)));
}

#if defined(DATA_A_Q4_K)
const uint QUANT_K = 256;
const uint SLICES_PER_SUBBLOCK = 2;
const uint SLICES_PER_BATCH = 4;
const uint HEADER_DWORDS = 4;

uint batch_slice(const uint batch, const uint j) {
    return batch * 4u + j;
}

const uint BLOCK_VEC4S   = 9;
const uint QS_FIRST_VEC4 = 1;
layout (binding = 0) readonly buffer A { uvec4 data_a_v4[]; };

void load_header(const uint block_index, const uint block_parity, const uint part, out uint header[HEADER_DWORDS]) {
    const uvec4 dm_scales = data_a_v4[block_index * BLOCK_VEC4S];
    [[unroll]] for (uint i = 0; i < 4u; ++i) header[i] = dm_scales[i];
}

vec2 subblock_scale_min(const uint header[HEADER_DWORDS], const uint slice) {
    const vec2 d_dmin = unpackHalf2x16(header[0]);
    const uint subblock = slice >> 1u;
    const int byte_shift = int((subblock & 3u) * 8u);
    const uint byte0 = bitfieldExtract(header[1], byte_shift, 8);
    const uint byte1 = bitfieldExtract(header[2], byte_shift, 8);
    const uint byte2 = bitfieldExtract(header[3], byte_shift, 8);
    const uint scale_low  = byte0 & 63u;
    const uint min_low    = byte1 & 63u;
    const uint scale_high = (byte2 & 15u) | ((byte0 >> 6u) << 4u);
    const uint min_high   = (byte2 >> 4u) | ((byte1 >> 6u) << 4u);
    const bool low = subblock < 4u;
    const float factor = (subblock & 1u) != 0u ? 1048576.0 : 16777216.0;
    return vec2(d_dmin.x * factor * float(low ? scale_low : scale_high), -d_dmin.y * float(low ? min_low : min_high));
}

const uint PAIR_DWORDS = 2;
const uint BLOCK_PAIR_DWORDS = 1;

void load_pair(const uint block_index0, const uint block_index1, const uint k0, const uint k1, const uint block_parity,
               const uint batch, out pair_t pair_data[PAIR_DWORDS]) {
    [[unroll]] for (uint r = 0; r < 2u; ++r) {
        const uint qs_index = QS_FIRST_VEC4 + batch * 2u + r;
        const uint dword0 = data_a_v4[block_index0 * BLOCK_VEC4S + qs_index][k0 >> 2u];
        const uint dword1 = data_a_v4[block_index1 * BLOCK_VEC4S + qs_index][k1 >> 2u];
        pair_data[r] = pack_byte_pair(dword0, dword1, k0, k1);
    }
}

void load_block_pair(const uint block_index0, const uint block_index1, const uint k0, const uint k1, const uint block_parity,
                     out pair_t block_data[BLOCK_PAIR_DWORDS]) {
    block_data[0] = PAIR_CONST(0u);
}

pair_t dequant_pair(const pair_t pair_data[PAIR_DWORDS], const pair_t block_data[BLOCK_PAIR_DWORDS], const uint batch, const uint j) {
    return pair_data[j & 1u] & PAIR_CONST(j >= 2u ? 0x00F0u : 0x000Fu);
}
#endif

#if defined(DATA_A_Q6_K)
const uint QUANT_K = 256;
const uint SLICES_PER_SUBBLOCK = 1;
const uint SLICES_PER_BATCH = 4;
const uint HEADER_DWORDS = 5;
const uint BLOCK_BYTES = 210;

uint batch_slice(const uint batch, const uint j) {
    return (batch >> 1u) * 8u + j * 2u + (batch & 1u);
}

layout (binding = 0) readonly buffer A { uint data_a_u32[]; };

const bool BLOCKS_MISALIGNED = (BLOCK_BYTES % 4u) != 0u;

uint block_first_dword(const uint block_index, const uint block_parity) {
    if (!BLOCKS_MISALIGNED) return block_index * (BLOCK_BYTES / 4u);
    const uint odd = block_parity == 2u ? block_index & 1u : block_parity;
    return (block_index >> 1u) * (BLOCK_BYTES / 2u) + odd * (BLOCK_BYTES / 4u);
}

uint pick_dword(const uint dwords[5], const uint i) {
    const uint dword0 = dwords[0], dword1 = dwords[1], dword2 = dwords[2], dword3 = dwords[3], dword4 = dwords[4];
    const uint dword01 = i == 1u ? dword1 : dword0;
    const uint dword23 = i == 3u ? dword3 : dword2;
    const uint dword03 = i >= 2u ? dword23 : dword01;
    return i >= 4u ? dword4 : dword03;
}

void load_row(const uint block_index, const uint block_parity, const uint row_offset, out uint dwords[5]) {
    const uint first_dword = block_first_dword(block_index, block_parity) + row_offset / 4u;
    [[unroll]] for (uint i = 0; i < 5u; ++i) dwords[i] = data_a_u32[first_dword + i];
    if (BLOCKS_MISALIGNED && block_parity == 2u) {
        const bool odd = (block_index & 1u) != 0u;
        [[unroll]] for (uint i = 0; i < 4u; ++i) {
            const uint aligned = dwords[i];
            const uint shifted = (dwords[i] >> 16u) | (dwords[i + 1u] << 16u);
            dwords[i] = odd ? shifted : aligned;
        }
    }
}

pair_t load_byte_pair(const uint block_index0, const uint block_index1, const uint block_parity, const uint row_offset,
                      const uint k0, const uint k1) {
    uint dwords0[5], dwords1[5];
    load_row(block_index0, block_parity, row_offset, dwords0);
    load_row(block_index1, block_parity, row_offset, dwords1);
    const uint start = BLOCKS_MISALIGNED && block_parity == 1u ? 2u : 0u;
    const uint position0 = start + k0;
    const uint position1 = start + k1;
    return pack_byte_pair(pick_dword(dwords0, position0 >> 2u), pick_dword(dwords1, position1 >> 2u), position0, position1);
}

void load_header(const uint block_index, const uint block_parity, const uint part, out uint header[HEADER_DWORDS]) {
    const uint first_dword = block_first_dword(block_index, block_parity) + 48u;
    const bool odd = block_parity == 2u ? (block_index & 1u) != 0u : block_parity == 1u;
    uint dwords[5];
    [[unroll]] for (uint i = 0; i < 5u; ++i) dwords[i] = data_a_u32[first_dword + i];
    [[unroll]] for (uint i = 0; i < 4u; ++i) {
        const uint aligned = dwords[i];
        const uint shifted = (dwords[i] >> 16u) | (dwords[i + 1u] << 16u);
        header[i] = odd ? shifted : aligned;
    }
    const uint d_aligned = dwords[4];
    const uint d_shifted = dwords[4] >> 16u;
    header[4] = odd ? d_shifted : d_aligned;
}

vec2 subblock_scale_min(const uint header[HEADER_DWORDS], const uint slice) {
    const float d = unpackHalf2x16(header[4]).x;
    const float scale = d * float(bitfieldExtract(int(pick_dword(header, slice >> 2u)), int((slice & 3u) * 8u), 8));
    const float factor = (slice & 4u) != 0u ? 1048576.0 : 16777216.0;
    return vec2(scale * factor, -32.0 * scale);
}

const uint PAIR_DWORDS = 3;
const uint BLOCK_PAIR_DWORDS = 1;

void load_block_pair(const uint block_index0, const uint block_index1, const uint k0, const uint k1, const uint block_parity,
                     out pair_t block_data[BLOCK_PAIR_DWORDS]) {
    block_data[0] = PAIR_CONST(0u);
}

void load_pair(const uint block_index0, const uint block_index1, const uint k0, const uint k1, const uint block_parity,
               const uint batch, out pair_t pair_data[PAIR_DWORDS]) {
    const uint half_index = batch >> 1u;
    const uint l_base     = (batch & 1u) * 16u;
    const uint ql_offset  = half_index * 64u + l_base;
    pair_data[0] = load_byte_pair(block_index0, block_index1, block_parity, ql_offset, k0, k1);
    pair_data[1] = load_byte_pair(block_index0, block_index1, block_parity, ql_offset + 32u, k0, k1);
    pair_data[2] = load_byte_pair(block_index0, block_index1, block_parity, 128u + half_index * 32u + l_base, k0, k1);
}

pair_t dequant_pair(const pair_t pair_data[PAIR_DWORDS], const pair_t block_data[BLOCK_PAIR_DWORDS], const uint batch, const uint j) {
    const pair_t ql = pair_data[j & 1u];
    const pair_t qh_shifted = pair_data[2] << ((j & 1u) != 0u ? 2u : 4u);
    return (ql & PAIR_CONST(j >= 2u ? 0x00F0u : 0x000Fu)) | (qh_shifted & PAIR_CONST(j >= 2u ? 0x0300u : 0x0030u));
}
#endif
