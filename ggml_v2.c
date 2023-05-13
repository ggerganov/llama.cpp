// quantization
//

#if __AVX__ || __AVX2__ || __AVX512F__
// Unpack 16 4-bit fields into 16 bytes
// The output vector contains 16 bytes, each one in [ 0 .. 15 ] interval
static inline __m128i bytes_from_nibbles_16(const uint8_t * rsi)
{
    // Load 8 bytes from memory
    __m128i tmp = _mm_loadl_epi64( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m128i bytes = _mm_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m128i lowMask = _mm_set1_epi8( 0xF );
    __m128i high = _mm_andnot_si128( lowMask, bytes );
    __m128i low = _mm_and_si128( lowMask, bytes );
    high = _mm_slli_epi16( high, 4 );
    bytes = _mm_or_si128( low, high );
    return bytes;
}



#if __AVX2__ || __AVX512F__

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32_v2(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128( ( const __m128i* )rsi );

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16( tmp );

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    __m256i high = _mm256_andnot_si256( lowMask, bytes );
    __m256i low = _mm256_and_si256( lowMask, bytes );
    high = _mm256_slli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );
    return bytes;
}
#endif
#endif


#if __ARM_NEON
#if !defined(__aarch64__)
int8x8_t vzip1_s8(int8x8_t a, int8x8_t b) {
    int8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

int8x8_t vzip2_s8(int8x8_t a, int8x8_t b) {
    int8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

int8x16_t vzip1q_s8(int8x16_t a, int8x16_t b) {
    int8x16_t res;

    res[0]  = a[0]; res[1]  = b[0]; res[2]  = a[1]; res[3]  = b[1];
    res[4]  = a[2]; res[5]  = b[2]; res[6]  = a[3]; res[7]  = b[3];
    res[8]  = a[4]; res[9]  = b[4]; res[10] = a[5]; res[11] = b[5];
    res[12] = a[6]; res[13] = b[6]; res[14] = a[7]; res[15] = b[7];

    return res;
}

int8x16_t vzip2q_s8(int8x16_t a, int8x16_t b) {
    int8x16_t res;

    res[0]  = a[8];  res[1]  = b[8];  res[2]  = a[9];  res[3]  = b[9];
    res[4]  = a[10]; res[5]  = b[10]; res[6]  = a[11]; res[7]  = b[11];
    res[8]  = a[12]; res[9]  = b[12]; res[10] = a[13]; res[11] = b[13];
    res[12] = a[14]; res[13] = b[14]; res[14] = a[15]; res[15] = b[15];

    return res;
}

uint8x16_t vzip1q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[0]  = a[0];  res[1]  = b[0];  res[2]  = a[1];  res[3]  = b[1];
    res[4]  = a[2];  res[5]  = b[2];  res[6]  = a[3];  res[7]  = b[3];
    res[8]  = a[4];  res[9]  = b[4];  res[10] = a[5];  res[11] = b[5];
    res[12] = a[6];  res[13] = b[6];  res[14] = a[7];  res[15] = b[7];

    return res;
}

uint8x16_t vzip2q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[0]  = a[8];  res[1]  = b[8];  res[2]  = a[9];  res[3]  = b[9];
    res[4]  = a[10]; res[5]  = b[10]; res[6]  = a[11]; res[7]  = b[11];
    res[8]  = a[12]; res[9]  = b[12]; res[10] = a[13]; res[11] = b[13];
    res[12] = a[14]; res[13] = b[14]; res[14] = a[15]; res[15] = b[15];

    return res;
}
#endif
#endif


// reference implementation for deterministic creation of model files
static void quantize_row_q4_0_reference_v2(const float * restrict x, block_q4_0 * restrict y, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    uint8_t pp[QK4_0/2];

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int l = 0; l < QK4_0; l++) {
            const float v = x[i*QK4_0 + l];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK4_0; l += 2) {
            const float v0 = x[i*QK4_0 + l + 0]*id;
            const float v1 = x[i*QK4_0 + l + 1]*id;

            const uint8_t vi0 = MIN(15, (int8_t)roundf(v0) + 8);
            const uint8_t vi1 = MIN(15, (int8_t)roundf(v1) + 8);

            assert(vi0 < 16);
            assert(vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
}

static void quantize_row_q4_0_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    block_q4_0 * restrict y = vy;

#if defined(__POWER9_VECTOR__)
    const vector float v85 = vec_splats(8.5f);
    const vector signed int v15 = vec_splats(15);
    for (int i = 0; i < nb; i++) {
        float max = 0.0f;
        float min = 0.0f;

        vector float asrcv [8];
        vector float srcv [8];
        vector float maxv[8];
        vector float minv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = *(vector float *)(x + i*32 + 4*l);
        //for (int l = 0; l < 8; l++) asrcv[l] = vec_abs(srcv[l]);

        for (int l = 0; l < 4; l++) maxv[2*l] = vec_max(asrcv[2*l], asrcv[2*l+1]);
        //for (int l = 0; l < 2; l++) maxv[4*l] = vec_max(maxv[4*l], maxv[4*l+2]);
        maxv[0] = vec_max(maxv[0], maxv[2]);
        maxv[4] = vec_max(maxv[4], maxv[6]);
        //for (int l = 0; l < 1; l++) maxv[8*l] = vec_max(maxv[8*l], maxv[8*l+4]);
        maxv[0] = vec_max(maxv[0], maxv[4]);

        for (int l = 0; l < 4; l++) minv[2*l] = vec_min(asrcv[2*l], asrcv[2*l+1]);
        //for (int l = 0; l < 2; l++) minv[4*l] = vec_min(minv[4*l], minv[4*l+2]);
        minv[0] = vec_min(minv[0], minv[2]);
        minv[4] = vec_min(minv[4], minv[6]);
        //for (int l = 0; l < 1; l++) minv[8*l] = vec_min(minv[8*l], minv[8*l+4]);
        minv[0] = vec_min(minv[0], minv[4]);


        max = MAX(
                MAX(vec_extract(maxv[0], 0), vec_extract(maxv[0], 1)),
                MAX(vec_extract(maxv[0], 2), vec_extract(maxv[0], 3)));
        min = MIN(
                MIN(vec_extract(minv[0], 0), vec_extract(minv[0], 1)),
                MIN(vec_extract(minv[0], 2), vec_extract(minv[0], 3)));

        const float magnitude = max >= fabsf(min) ? max : min;
        const float d = magnitude / -8;
        const float id = d ? 1.0/d : 0.0;

        y[i].d = d;

        const vector float vid = vec_splats(id);
        uint8_t * restrict pb = y[i].qs;
        for (int l = 0; l < 8; l++) {
            const vector float vf  = vec_madd(srcv[l], vid, v85);
            const vector signed int vi = vec_signed(vf);
            const vector signed int vc = vec_min(vi, v15);

            pb[2*l + 0] = vec_extract(vc, 0) | (vec_extract(vc, 1) << 4);
            pb[2*l + 1] = vec_extract(vc, 2) | (vec_extract(vc, 3) << 4);
        }
    }
#elif __ARM_NEON
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t maxv[8];
        float32x4_t minv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);

        for (int l = 0; l < 4; l++) maxv[2*l] = vmaxq_f32(srcv[2*l], srcv[2*l+1]);
        for (int l = 0; l < 2; l++) maxv[4*l] = vmaxq_f32(maxv[4*l], maxv[4*l+2]);
        for (int l = 0; l < 1; l++) maxv[8*l] = vmaxq_f32(maxv[8*l], maxv[8*l+4]);

        for (int l = 0; l < 4; l++) minv[2*l] = vminq_f32(srcv[2*l], srcv[2*l+1]);
        for (int l = 0; l < 2; l++) minv[4*l] = vminq_f32(minv[4*l], minv[4*l+2]);
        for (int l = 0; l < 1; l++) minv[8*l] = vminq_f32(minv[8*l], minv[8*l+4]);

        const float max = vmaxvq_f32(maxv[0]);
        const float min = vminvq_f32(minv[0]);

        const float magnitude = max >= fabsf(min) ? max : min;
        const float d = magnitude / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(8.5f));
            const int32x4_t   vi = vcvtq_s32_f32(vf);
            const int32x4_t   vc = vminq_s32(vi, vdupq_n_s32(15));

            y[i].qs[2*l + 0] = vgetq_lane_s32(vc, 0) | (vgetq_lane_s32(vc, 1) << 4);
            y[i].qs[2*l + 1] = vgetq_lane_s32(vc, 2) | (vgetq_lane_s32(vc, 3) << 4);
        }
    }
#elif defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max for the block
        __m256 max  = _mm256_max_ps( v0, v1 );
        __m256 maxTmp = _mm256_max_ps( v2, v3 );
        max = _mm256_max_ps( max, maxTmp );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( max, 1 ), _mm256_castps256_ps128( max ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Compute min for the block
        __m256 min  = _mm256_min_ps( v0, v1 );
        __m256 minTmp = _mm256_min_ps( v2, v3 );
        min = _mm256_min_ps( min, minTmp );

        __m128 min4 = _mm_min_ps( _mm256_extractf128_ps( min, 1 ), _mm256_castps256_ps128( min ) );
        min4 = _mm_min_ps( min4, _mm_movehl_ps( min4, min4 ) );
        min4 = _mm_min_ss( min4, _mm_movehdup_ps( min4 ) );
        const float minScalar = _mm_cvtss_f32( min4 );

        // Quantize these floats
        const float magnitude = maxScalar >= fabsf(minScalar) ? maxScalar : minScalar;
        const float d = magnitude / -8.0f;
        y[i].d = d;
        const float id = ( magnitude != 0.0f ) ? -8.0f / magnitude : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Apply offset and clamp to translate the range from [ -8 .. +8 ] into [ +0 .. +15 ]
        const __m256i off = _mm256_set1_epi8( 8 );
        i0 = _mm256_add_epi8( i0, off );
        const __m256i maxNibble = _mm256_set1_epi8( 15 );
        i0 = _mm256_min_epi8( i0, maxNibble );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max for the block
        __m256 max  = _mm256_max_ps( v0, v1 );
        __m256 maxTmp = _mm256_max_ps( v2, v3 );
        max = _mm256_max_ps( max, maxTmp );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( max, 1 ), _mm256_castps256_ps128( max ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Compute min for the block
        __m256 min  = _mm256_min_ps( v0, v1 );
        __m256 minTmp = _mm256_min_ps( v2, v3 );
        min = _mm256_min_ps( min, minTmp );

        __m128 min4 = _mm_min_ps( _mm256_extractf128_ps( min, 1 ), _mm256_castps256_ps128( min ) );
        min4 = _mm_min_ps( min4, _mm_movehl_ps( min4, min4 ) );
        min4 = _mm_min_ss( min4, _mm_movehdup_ps( min4 ) );
        const float minScalar = _mm_cvtss_f32( min4 );

        // Quantize these floats
        const float magnitude = maxScalar >= fabsf(minScalar) ? maxScalar : minScalar;
        const float d = magnitude / -8.0f;
        y[i].d = d;
        const float id = ( magnitude != 0.0f ) ? -8.0f / magnitude : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        // Apply offset and clamp to translate the range from [ -8 .. +8 ] into [ +0 .. +15 ]
        const __m128i off = _mm_set1_epi8( 8 );
        ni0 = _mm_add_epi8( ni0, off );
        ni4 = _mm_add_epi8( ni4, off );
        const __m128i maxNibble = _mm_set1_epi8( 15 );
        ni0 = _mm_min_epi8( ni0, maxNibble );
        ni4 = _mm_min_epi8( ni4, maxNibble );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( ni0, ni4 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        float max = 0.0f;
        float min = 0.0f;

        v128_t srcv [8];
        v128_t maxv[8];
        v128_t minv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = wasm_v128_load(x + i*32 + 4*l);

        for (int l = 0; l < 4; l++) maxv[2*l] = wasm_f32x4_max(srcv[2*l], srcv[2*l+1]);
        for (int l = 0; l < 2; l++) maxv[4*l] = wasm_f32x4_max(maxv[4*l], maxv[4*l+2]);
        for (int l = 0; l < 1; l++) maxv[8*l] = wasm_f32x4_max(maxv[8*l], maxv[8*l+4]);

        for (int l = 0; l < 4; l++) minv[2*l] = wasm_f32x4_min(srcv[2*l], srcv[2*l+1]);
        for (int l = 0; l < 2; l++) minv[4*l] = wasm_f32x4_min(minv[4*l], minv[4*l+2]);
        for (int l = 0; l < 1; l++) minv[8*l] = wasm_f32x4_min(minv[8*l], minv[8*l+4]);

        max = MAX(
                MAX(wasm_f32x4_extract_lane(maxv[0], 0), wasm_f32x4_extract_lane(maxv[0], 1)),
                MAX(wasm_f32x4_extract_lane(maxv[0], 2), wasm_f32x4_extract_lane(maxv[0], 3)));
        min = MIN(
                MIN(wasm_f32x4_extract_lane(minv[0], 0), wasm_f32x4_extract_lane(minv[0], 1)),
                MIN(wasm_f32x4_extract_lane(minv[0], 2), wasm_f32x4_extract_lane(minv[0], 3)));

        const float magnitude = max >= fabsf(min) ? max : min;
        const float d = magnitude / -8;
        const float id = d ? 1.0/d : 0.0;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const v128_t v  = wasm_f32x4_mul(srcv[l], wasm_f32x4_splat(id));
            const v128_t vf = wasm_f32x4_add(v, wasm_f32x4_splat(8.5f));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(vf);
            const v128_t vc = wasm_i32x4_min(vi, wasm_i32x4_splat(15));

            y[i].qs[2*l + 0] = wasm_i32x4_extract_lane(vc, 0) | (wasm_i32x4_extract_lane(vc, 1) << 4);
            y[i].qs[2*l + 1] = wasm_i32x4_extract_lane(vc, 2) | (wasm_i32x4_extract_lane(vc, 3) << 4);
        }
    }
#else
    // scalar
    quantize_row_q4_0_reference_v2(x, y, k);
#endif
}

static void quantize_row_q4_1_reference_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    block_q4_1 * restrict y = vy;

    uint8_t pp[QK4_1/2];

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK4_1; l++) {
            const float v = x[i*QK4_1 + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;
        y[i].m = min;

        for (int l = 0; l < QK4_1; l += 2) {
            const float v0 = (x[i*QK4_1 + l + 0] - min)*id;
            const float v1 = (x[i*QK4_1 + l + 1] - min)*id;

            const uint8_t vi0 = roundf(v0);
            const uint8_t vi1 = roundf(v1);

            assert(vi0 < 16);
            assert(vi1 < 16);

            pp[l/2] = vi0 | (vi1 << 4);
        }

        memcpy(y[i].qs, pp, sizeof(pp));
    }
}

static void quantize_row_q4_1_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_1 == 0);

    const int nb = k / QK4_1;

    block_q4_1 * restrict y = vy;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max for the block
        __m256 vmax;
        vmax = _mm256_max_ps( v0, v1 );
        vmax = _mm256_max_ps( vmax, v2 );
        vmax = _mm256_max_ps( vmax, v3 );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( vmax, 1 ), _mm256_castps256_ps128( vmax ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Compute min for the block
        __m256 vmin;
        vmin = _mm256_min_ps( v0, v1 );
        vmin = _mm256_min_ps( vmin, v2 );
        vmin = _mm256_min_ps( vmin, v3 );

        __m128 min4 = _mm_min_ps( _mm256_extractf128_ps( vmin, 1 ), _mm256_castps256_ps128( vmin ) );
        min4 = _mm_min_ps( min4, _mm_movehl_ps( min4, min4 ) );
        min4 = _mm_min_ss( min4, _mm_movehdup_ps( min4 ) );
        const float minScalar = _mm_cvtss_f32( min4 );

        // Quantize these floats
        const float d = (maxScalar - minScalar) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].m = minScalar;
        y[i].d = d;

        // x = (x-min)*id
        const __m256 mul = _mm256_set1_ps( id );
        const __m256 off = _mm256_set1_ps( minScalar );
        v0 = _mm256_mul_ps( _mm256_sub_ps( v0, off ), mul );
        v1 = _mm256_mul_ps( _mm256_sub_ps( v1, off ), mul );
        v2 = _mm256_mul_ps( _mm256_sub_ps( v2, off ), mul );
        v3 = _mm256_mul_ps( _mm256_sub_ps( v3, off ), mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        // Compress the vector into 4 bit/value, and store
        __m128i res = packNibbles( i0 );
        _mm_storeu_si128( ( __m128i* )y[i].qs, res );
    }
#elif __ARM_NEON
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv[8];
        float32x4_t minv[8];
        float32x4_t maxv[8];

        for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(x + i*QK4_1 + 4*l);

        for (int l = 0; l < 4; l++) minv[2*l] = vminq_f32(srcv[2*l], srcv[2*l + 1]);
        for (int l = 0; l < 2; l++) minv[4*l] = vminq_f32(minv[4*l], minv[4*l + 2]);
        for (int l = 0; l < 1; l++) minv[8*l] = vminq_f32(minv[8*l], minv[8*l + 4]);

        for (int l = 0; l < 4; l++) maxv[2*l] = vmaxq_f32(srcv[2*l], srcv[2*l + 1]);
        for (int l = 0; l < 2; l++) maxv[4*l] = vmaxq_f32(maxv[4*l], maxv[4*l + 2]);
        for (int l = 0; l < 1; l++) maxv[8*l] = vmaxq_f32(maxv[8*l], maxv[8*l + 4]);

        const float min = vminvq_f32(minv[0]);
        const float max = vmaxvq_f32(maxv[0]);

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;
        y[i].m = min;

        const float32x4_t minv0 = vdupq_n_f32(min);

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(vsubq_f32(srcv[l], minv0), id);
            const float32x4_t vf = vaddq_f32(v, vdupq_n_f32(0.5f)); // needed to round to nearest
            const int32x4_t   vi = vcvtq_s32_f32(vf);

            y[i].qs[2*l + 0] = vgetq_lane_s32(vi, 0) | (vgetq_lane_s32(vi, 1) << 4);
            y[i].qs[2*l + 1] = vgetq_lane_s32(vi, 2) | (vgetq_lane_s32(vi, 3) << 4);
        }
    }
#else
    // scalar
    quantize_row_q4_1_reference_v2(x, vy, k);
#endif
}

// reference implementation for deterministic creation of model files
static void quantize_row_q4_2_reference_v2(const float * restrict x, block_q4_2 * restrict y, int k) {
    assert(k % QK4_2 == 0);

    const int nb = k / QK4_2;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int l = 0; l < QK4_2; l++) {
            const float v = x[i*QK4_2 + l];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;

        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int l = 0; l < QK4_2; l += 2) {
            const float v0 = x[i*QK4_2 + l + 0]*id;
            const float v1 = x[i*QK4_2 + l + 1]*id;

            const uint8_t vi0 = MIN(15, (uint8_t)(v0 + 8.5f));
            const uint8_t vi1 = MIN(15, (uint8_t)(v1 + 8.5f));

            assert(vi0 < 16);
            assert(vi1 < 16);

            y[i].qs[l/2] = vi0 | (vi1 << 4);
        }
    }
}

static void quantize_row_q4_2_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_2 == 0);

    block_q4_2 * restrict y = vy;

    quantize_row_q4_2_reference_v2(x, y, k);
}

static void quantize_row_q4_3_reference_v2(const float * restrict x, block_q4_3 * restrict y, int k) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK4_3; l++) {
            const float v = x[i*QK4_3 + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        for (int l = 0; l < QK4_3; l += 2) {
            const float v0 = (x[i*QK4_3 + l + 0] - min)*id;
            const float v1 = (x[i*QK4_3 + l + 1] - min)*id;

            const uint8_t vi0 = (int) (v0 + 0.5f);
            const uint8_t vi1 = (int) (v1 + 0.5f);

            assert(vi0 < 16);
            assert(vi1 < 16);

            y[i].qs[l/2] = vi0 | (vi1 << 4);
        }
    }
}

static void quantize_row_q4_3_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK4_3 == 0);

    block_q4_3 * restrict y = vy;

    quantize_row_q4_3_reference_v2(x, y, k);
}

static void quantize_row_q5_0_reference_v2(const float * restrict x, block_q5_0 * restrict y, int k) {
    assert(k % QK5_0 == 0);
    const int nb = k / QK5_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int l = 0; l < QK5_0; l++) {
            const float v = x[i*QK5_0 + l];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        uint32_t qh = 0;

        for (int l = 0; l < QK5_0; l += 2) {
            const float v0 = x[i*QK5_0 + l + 0]*id;
            const float v1 = x[i*QK5_0 + l + 1]*id;

            const uint32_t vi0 = MIN(31, (int) (v0 + 16.5f));
            const uint32_t vi1 = MIN(31, (int) (v1 + 16.5f));

            y[i].qs[l/2] = (vi0 & 0x0F) | ((vi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((vi0 & 0x10) >> 4) << (l + 0);
            qh |= ((vi1 & 0x10) >> 4) << (l + 1);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

static void quantize_row_q5_0_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK5_0 == 0);

    block_q5_0 * restrict y = vy;

    quantize_row_q5_0_reference_v2(x, y, k);
}

static void quantize_row_q5_1_reference_v2(const float * restrict x, block_q5_1 * restrict y, int k) {
    assert(k % QK5_1 == 0);
    const int nb = k / QK5_1;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int l = 0; l < QK5_1; l++) {
            const float v = x[i*QK5_1 + l];
            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(min);

        uint32_t qh = 0;

        for (int l = 0; l < QK5_1; l += 2) {
            const float v0 = (x[i*QK5_1 + l + 0] - min)*id;
            const float v1 = (x[i*QK5_1 + l + 1] - min)*id;

            const uint32_t vi0 = (int) (v0 + 0.5f);
            const uint32_t vi1 = (int) (v1 + 0.5f);

            y[i].qs[l/2] = (vi0 & 0x0F) | ((vi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((vi0 & 0x10) >> 4) << (l + 0);
            qh |= ((vi1 & 0x10) >> 4) << (l + 1);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

static void quantize_row_q5_1_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK5_1 == 0);

    block_q5_1 * restrict y = vy;

    quantize_row_q5_1_reference_v2(x, y, k);
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_0_reference_v2(const float * restrict x, block_q8_0 * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK8_0; l++) {
            const float v = x[i*QK8_0 + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < QK8_0; ++l) {
            const float v0 = x[i*QK8_0 + l]*id;

            y[i].qs[l] = roundf(v0);
        }
    }
}

static void quantize_row_q8_0_v2(const float * restrict x, void * restrict vy, int k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        for (int l = 0; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*l + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*l + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*l + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*l + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    // scalar
    quantize_row_q8_0_reference_v2(x, y, k);
#endif
}

// reference implementation for deterministic creation of model files
static void quantize_row_q8_1_reference_v2(const float * restrict x, block_q8_1_v2 * restrict y, int k) {
    assert(QK8_1 == 32);
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int l = 0; l < QK8_1; l++) {
            const float v = x[i*QK8_1 + l];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        int sum0 = 0;
        int sum1 = 0;

        for (int l = 0; l < QK8_1/2; ++l) {
            const float v0 = x[i*QK8_1           + l]*id;
            const float v1 = x[i*QK8_1 + QK8_1/2 + l]*id;

            y[i].qs[          l] = roundf(v0);
            y[i].qs[QK8_1/2 + l] = roundf(v1);

            sum0 += y[i].qs[          l];
            sum1 += y[i].qs[QK8_1/2 + l];
        }

        y[i].s0 = d * sum0;
        y[i].s1 = d * sum1;
    }
}

static void quantize_row_q8_1_v2(const float * restrict x, void * restrict vy, int k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1_v2 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int l = 0; l < 8; l++) srcv[l]  = vld1q_f32(x + i*32 + 4*l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2*l] = vmaxq_f32(asrcv[2*l], asrcv[2*l+1]);
        for (int l = 0; l < 2; l++) amaxv[4*l] = vmaxq_f32(amaxv[4*l], amaxv[4*l+2]);
        for (int l = 0; l < 1; l++) amaxv[8*l] = vmaxq_f32(amaxv[8*l], amaxv[8*l+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = d;

        int32x4_t accv0 = vdupq_n_s32(0);
        int32x4_t accv1 = vdupq_n_s32(0);

        // low half
        for (int l = 0; l < 4; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*l + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*l + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*l + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*l + 3] = vgetq_lane_s32(vi, 3);

            accv0 = vaddq_s32(accv0, vi);
        }

        // high half
        for (int l = 4; l < 8; l++) {
            const float32x4_t v  = vmulq_n_f32(srcv[l], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*l + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*l + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*l + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*l + 3] = vgetq_lane_s32(vi, 3);

            accv1 = vaddq_s32(accv1, vi);
        }

        const int32_t sum0 = vaddvq_s32(accv0);
        const int32_t sum1 = vaddvq_s32(accv1);

        y[i].s0 = d * sum0;
        y[i].s1 = d * sum1;
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        //y[i].s = d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        y[i].s0 = d * hsum_i32_8(_mm256_add_epi32(i0, i1));
        y[i].s1 = d * hsum_i32_8(_mm256_add_epi32(i2, i3));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s0 = d * hsum_i32_4(s0);
        y[i].s1 = d * hsum_i32_4(s1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    // scalar
    quantize_row_q8_1_reference_v2(x, y, k);
#endif
}

static void dequantize_row_q4_0_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    const block_q4_0 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        // scale factor
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32_v2(pp+l/2);

            // Subtract 8 from the integers
            vx8 = _mm256_sub_epi8(vx8, _mm256_set1_epi8(8));

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_mul_ps(vf[j], d_v);
                _mm256_storeu_ps(y + i * QK4_0 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0F));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Convert to signed 8-bit integers
            const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
            const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

            // Subtract 8 from each byte
            const int8x8_t vb_0 = vsub_s8(vs_0, vdup_n_s8(8));
            const int8x8_t vb_1 = vsub_s8(vs_1, vdup_n_s8(8));

            // Interleave and combine
            const int8x8_t vx_0 = vzip1_s8(vb_0, vb_1);
            const int8x8_t vx_1 = vzip2_s8(vb_0, vb_1);

            const int8x16_t vq = vcombine_s8(vx_0, vx_1);

            // convert to 2x int16x8_t
            const int16x8_t vi_0 = vmovl_s8(vget_low_s8 (vq));
            const int16x8_t vi_1 = vmovl_s8(vget_high_s8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_1)));

            // Multiply by d
            const float32x4_t r0 = vmulq_f32(vf_0, vd);
            const float32x4_t r1 = vmulq_f32(vf_1, vd);
            const float32x4_t r2 = vmulq_f32(vf_2, vd);
            const float32x4_t r3 = vmulq_f32(vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_0 + l +  0, r0);
            vst1q_f32(y + i*QK4_0 + l +  4, r1);
            vst1q_f32(y + i*QK4_0 + l +  8, r2);
            vst1q_f32(y + i*QK4_0 + l + 12, r3);
        }
    }
#else
    // scalar
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_0; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            //printf("d = %f, vi = %d, vi0 = %d, vi1 = %d, v0 = %f, v1 = %f\n", d, vi, vi0, vi1, v0, v1);

            y[i*QK4_0 + l + 0] = v0;
            y[i*QK4_0 + l + 1] = v1;

            assert(!isnan(y[i*QK4_0 + l + 0]));
            assert(!isnan(y[i*QK4_0 + l + 1]));
        }
    }
#endif
}

static void dequantize_row_q4_1_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    const block_q4_1 * restrict x = vx;

#if defined(__AVX2__)
    for (int i = 0; i < nb; i++) {
        const __m256 d_v = _mm256_broadcast_ss(&x[i].d);
        const __m256 d_m = _mm256_broadcast_ss(&x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 32) {
            // Load 32x4-bit integers into 32x8-bit integers
            __m256i vx8 = bytes_from_nibbles_32_v2(pp+l/2);

            // Convert to 16-bit int
            const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
            const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

            // Convert to 32-bit int -> float 32
            const __m256 vf[4] = {
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))
            };

            // Scale, add m and store
            for (int j = 0; j < 4; j++) {
                const __m256 result = _mm256_add_ps(_mm256_mul_ps(vf[j], d_v), d_m);
                _mm256_storeu_ps(y + i * QK4_1 + l + j*8, result);
            }
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float32x4_t vd = vdupq_n_f32(x[i].d);
        const float32x4_t vm = vdupq_n_f32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 16) {
            // Load 16x4-bit integers into 8x8-bit integers
            const uint8x8_t v8 = vld1_u8(pp + l/2);

            // Expand 4-bit qs to 8-bit bytes
            const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0F));
            const uint8x8_t v1 = vshr_n_u8(v8, 4);

            // Interleave and combine
            const uint8x8_t vx_0 = vzip1_u8(v0, v1);
            const uint8x8_t vx_1 = vzip2_u8(v0, v1);

            const uint8x16_t vq = vcombine_u8(vx_0, vx_1);

            // convert to 2x uint16x8_t
            const uint16x8_t vi_0 = vmovl_u8(vget_low_u8 (vq));
            const uint16x8_t vi_1 = vmovl_u8(vget_high_u8(vq));

            // convert to 4x float32x4_t
            const float32x4_t vf_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_0)));
            const float32x4_t vf_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_0)));
            const float32x4_t vf_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16 (vi_1)));
            const float32x4_t vf_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vi_1)));

            // multiply by d and add m
            const float32x4_t r0 = vmlaq_f32(vm, vf_0, vd);
            const float32x4_t r1 = vmlaq_f32(vm, vf_1, vd);
            const float32x4_t r2 = vmlaq_f32(vm, vf_2, vd);
            const float32x4_t r3 = vmlaq_f32(vm, vf_3, vd);

            // Store
            vst1q_f32(y + i*QK4_1 + l +  0, r0);
            vst1q_f32(y + i*QK4_1 + l +  4, r1);
            vst1q_f32(y + i*QK4_1 + l +  8, r2);
            vst1q_f32(y + i*QK4_1 + l + 12, r3);
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;
        const float m = x[i].m;

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_1; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK4_1 + l + 0] = v0;
            y[i*QK4_1 + l + 1] = v1;

            assert(!isnan(y[i*QK4_1 + l + 0]));
            assert(!isnan(y[i*QK4_1 + l + 1]));
        }
    }
#endif
}

static void dequantize_row_q4_2_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_2 == 0);
    const int nb = k / QK4_2;

    const block_q4_2 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_2; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = (vi0 - 8)*d;
            const float v1 = (vi1 - 8)*d;

            y[i*QK4_2 + l + 0] = v0;
            y[i*QK4_2 + l + 1] = v1;

            assert(!isnan(y[i*QK4_2 + l + 0]));
            assert(!isnan(y[i*QK4_2 + l + 1]));
        }
    }
}

static void dequantize_row_q4_3_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    const block_q4_3 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK4_3; l += 2) {
            const uint8_t vi = pp[l/2];

            const int8_t vi0 = vi & 0x0F;
            const int8_t vi1 = vi >> 4;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK4_3 + l + 0] = v0;
            y[i*QK4_3 + l + 1] = v1;

            assert(!isnan(y[i*QK4_3 + l + 0]));
            assert(!isnan(y[i*QK4_3 + l + 1]));
        }
    }
}

static void dequantize_row_q5_0_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK5_0 == 0);
    const int nb = k / QK5_0;

    const block_q5_0 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict pp = x[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int l = 0; l < QK5_0; l += 2) {
            const uint8_t vi = pp[l/2];

            // extract the 5-th bit from qh
            const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
            const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

            const int8_t vi0 = (vi & 0x0F) | vh0;
            const int8_t vi1 = (vi >>   4) | vh1;

            const float v0 = (vi0 - 16)*d;
            const float v1 = (vi1 - 16)*d;

            y[i*QK5_0 + l + 0] = v0;
            y[i*QK5_0 + l + 1] = v1;

            assert(!isnan(y[i*QK5_0 + l + 0]));
            assert(!isnan(y[i*QK5_0 + l + 1]));
        }
    }
}

static void dequantize_row_q5_1_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK5_1 == 0);
    const int nb = k / QK5_1;

    const block_q5_1 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        const uint8_t * restrict pp = x[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int l = 0; l < QK5_1; l += 2) {
            const uint8_t vi = pp[l/2];

            // extract the 5-th bit from qh
            const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
            const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

            const uint8_t vi0 = (vi & 0x0F) | vh0;
            const uint8_t vi1 = (vi >>   4) | vh1;

            const float v0 = vi0*d + m;
            const float v1 = vi1*d + m;

            y[i*QK5_1 + l + 0] = v0;
            y[i*QK5_1 + l + 1] = v1;

            assert(!isnan(y[i*QK5_1 + l + 0]));
            assert(!isnan(y[i*QK5_1 + l + 1]));
        }
    }
}

static void dequantize_row_q8_0_v2(const void * restrict vx, float * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    const block_q8_0 * restrict x = vx;

    for (int i = 0; i < nb; i++) {
        const float d = x[i].d;

        const int8_t * restrict pp = x[i].qs;

        for (int l = 0; l < QK8_0; ++l) {
            y[i*QK8_0 + l] = pp[l]*d;
        }
    }
}

static void ggml_vec_dot_q4_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_1_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_2_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q4_3_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q5_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q5_1_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);
static void ggml_vec_dot_q8_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy);

void SetQuantsUnshuffled(bool unshuffle)
{
    quants_unshuffled = unshuffle;
}

//TODO: integrate backwards compat
static const quantize_fns_t quantize_fns_v2[GGML_TYPE_COUNT] = {
    [GGML_TYPE_Q4_0] = {
        .dequantize_row_q         = dequantize_row_q4_0_v2,
        .quantize_row_q           = quantize_row_q4_0_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_0_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_0_v2,
        .vec_dot_q                = ggml_vec_dot_q4_0_q8_0_v2,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q4_1] = {
        .dequantize_row_q         = dequantize_row_q4_1_v2,
        .quantize_row_q           = quantize_row_q4_1_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_1_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_1_v2,
        .vec_dot_q                = ggml_vec_dot_q4_1_q8_1_v2,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q4_2] = {
        .dequantize_row_q         = dequantize_row_q4_2_v2,
        .quantize_row_q           = quantize_row_q4_2_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_2_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_0_v2,
        .vec_dot_q                = ggml_vec_dot_q4_2_q8_0_v2,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q4_3] = {
        .dequantize_row_q         = dequantize_row_q4_3_v2,
        .quantize_row_q           = quantize_row_q4_3_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q4_3_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_1_v2,
        .vec_dot_q                = ggml_vec_dot_q4_3_q8_1_v2,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q5_0] = {
        .dequantize_row_q         = dequantize_row_q5_0_v2,
        .quantize_row_q           = quantize_row_q5_0_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q5_0_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_0_v2,
        .vec_dot_q                = ggml_vec_dot_q5_0_q8_0_v2,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q5_1] = {
        .dequantize_row_q         = dequantize_row_q5_1_v2,
        .quantize_row_q           = quantize_row_q5_1_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q5_1_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_1_v2,
        .vec_dot_q                = ggml_vec_dot_q5_1_q8_1_v2,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q8_0] = {
        .dequantize_row_q         = dequantize_row_q8_0_v2,
        .quantize_row_q           = quantize_row_q8_0_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q8_0_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_0_v2,
        .vec_dot_q                = ggml_vec_dot_q8_0_q8_0_v2,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q8_1] = {
        .dequantize_row_q         = NULL,   // TODO
        .quantize_row_q           = quantize_row_q8_1_v2,
        .quantize_row_q_reference = (quantize_row_q_t) quantize_row_q8_1_reference_v2,
        .quantize_row_q_dot       = quantize_row_q8_1_v2,
        .vec_dot_q                = NULL,   // TODO
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
};


static void ggml_vec_dot_q4_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_0 * restrict x0 = &x[i + 0];
        const block_q4_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b   = vdupq_n_u8(0x0F);
        const int8x16_t  s8b   = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_1lz = vzip1q_s8(v0_1ls, v0_1hs);
        const int8x16_t v0_1hz = vzip2q_s8(v0_1ls, v0_1hs);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l), v0_0hz, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1lz, v1_1l), v0_1hz, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lz), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lz), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hz), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hz), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), x1->d*y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m256i bx = bytes_from_nibbles_32_v2(x[i].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        bx = _mm256_sub_epi8( bx, off );

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );

        __m128i i32[2];
        for (int j = 0; j < 2; ++j) {
            // Load 8 bytes, and unpack 4 bit fields into bytes, making 16 bytes
            __m128i bx = bytes_from_nibbles_16(x[i].qs + 8*j);
            __m128i by = _mm_loadu_si128((const __m128i *)(y[i].qs + 16*j));

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m128i off = _mm_set1_epi8( 8 );
            bx = _mm_sub_epi8( bx, off );

            // Get absolute values of x vectors
            const __m128i ax = _mm_sign_epi8(bx, bx);

            // Sign the values of the y vectors
            const __m128i sy = _mm_sign_epi8(by, bx);

            // Perform multiplication and create 16-bit values
            const __m128i dot = _mm_maddubs_epi16(ax, sy);

            const __m128i ones = _mm_set1_epi16(1);
            i32[j] = _mm_madd_epi16(ones, dot);
        }

        // Convert int32_t to float
        __m256 p = _mm256_cvtepi32_ps( _mm256_set_m128i( i32[0], i32[1] ));
        // Apply the scale, and accumulate
        acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
    }

    *s = hsum_float_8(acc);
#else
    // scalar
    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float d1 = y[i].d;

        const uint8_t * restrict p0 = x[i].qs;
        const  int8_t * restrict p1 = y[i].qs;

        int sumi = 0;
        for (int j = 0; j < QK8_0/2; j++) {
            const uint8_t v0 = p0[j];

            const int i0 = (int8_t) (v0 & 0x0F) - 8;
            const int i1 = (int8_t) (v0 >>   4) - 8;

            const int i2 = p1[2*j + 0];
            const int i3 = p1[2*j + 1];

            sumi += i0*i2 + i1*i3;
        }
        sumf += d0*d1*sumi;
    }
    *s = sumf;
#endif
}

static void ggml_vec_dot_q4_1_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_1;

    assert(n % QK8_1 == 0);
    assert(nb % 2 == 0);

    const block_q4_1 * restrict x = vx;
    const block_q8_1_v2 * restrict y = vy;

    // TODO: add AVX / WASM SIMD / etc
#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    for (int i = 0; i < nb; i += 2) {
        const block_q4_1 * restrict x0 = &x[i + 0];
        const block_q4_1 * restrict x1 = &x[i + 1];
        const block_q8_1_v2 * restrict y0 = &y[i + 0];
        const block_q8_1_v2 * restrict y1 = &y[i + 1];

        summs += x0->m * (y0->s0 + y0->s1) + x1->m * (y1->s0 + y1->s1);

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0l, v0_0h);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0l, v0_0h);
        const int8x16_t v0_1lz = vzip1q_s8(v0_1l, v0_1h);
        const int8x16_t v0_1hz = vzip2q_s8(v0_1l, v0_1h);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        // dot product into int32x4_t
        const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l), v0_0hz, v1_0h);
        const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1lz, v1_1l), v0_1hz, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), x1->d*y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lz), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lz), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hz), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hz), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), x1->d*y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    // Main loop
    for (int i = 0; i < nb; ++i) {
        const float * d0 = &x[i].d;
        const float * d1 = &y[i].d;

        summs += x[i].m * (y[i].s0 + y[i].s1);

        const __m256 d0v = _mm256_broadcast_ss( d0 );
        const __m256 d1v = _mm256_broadcast_ss( d1 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i bx = bytes_from_nibbles_32_v2(x[i].qs);
        const __m256i by = _mm256_loadu_si256( (const __m256i *)y[i].qs );

        const __m256 xy = mul_sum_i8_pairs_float(bx, by);

        // Accumulate d0*d1*x*y
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
    }

    *s = hsum_float_8(acc) + summs;
#else
    // scalar
    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const float d0 = x[i].d;
        const float m0 = x[i].m;
        const float d1 = y[i].d;

        const uint8_t * restrict p0 = x[i].qs;
        const  int8_t * restrict p1 = y[i].qs;

        // TODO: this is very slow ..
        for (int j = 0; j < QK8_1/2; j++) {
            const uint8_t v0 = p0[j];

            const float f0 = d0*(v0 & 0x0F) + m0;
            const float f1 = d0*(v0 >>   4) + m0;

            const float f2 = d1*p1[2*j + 0];
            const float f3 = d1*p1[2*j + 1];

            sumf += f0*f2 + f1*f3;
        }
    }
    *s = sumf;
#endif
}

static void ggml_vec_dot_q4_2_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);
    assert(QK8_0 == 2*QK4_2);

    const block_q4_2 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q4_2 * restrict x0_0 = &x[2*(i + 0) + 0];
        const block_q4_2 * restrict x0_1 = &x[2*(i + 0) + 1];
        const block_q4_2 * restrict x1_0 = &x[2*(i + 1) + 0];
        const block_q4_2 * restrict x1_1 = &x[2*(i + 1) + 1];

        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b   = vdupq_n_u8(0x0F);
        const int8x16_t  s8b   = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vcombine_u8(vld1_u8(x0_0->qs), vld1_u8(x0_1->qs));
        const uint8x16_t v0_1 = vcombine_u8(vld1_u8(x1_0->qs), vld1_u8(x1_1->qs));

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0ls, v0_0hs);
        const int8x16_t v0_1lz = vzip1q_s8(v0_1ls, v0_1hs);
        const int8x16_t v0_1hz = vzip2q_s8(v0_1ls, v0_1hs);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l)), GGML_FP16_TO_FP32(x0_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0hz, v1_0h)), GGML_FP16_TO_FP32(x0_1->d))), y0->d);

        sumv1 = vmlaq_n_f32(sumv1, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1lz, v1_1l)), GGML_FP16_TO_FP32(x1_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_1hz, v1_1h)), GGML_FP16_TO_FP32(x1_1->d))), y1->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1lz), vget_low_s8 (v1_1l));
        const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1lz), vget_high_s8(v1_1l));
        const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hz), vget_low_s8 (v1_1h));
        const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hz), vget_high_s8(v1_1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
        const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
        const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

        sumv0 = vmlaq_n_f32(sumv0, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(pl0), GGML_FP16_TO_FP32(x0_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(ph0), GGML_FP16_TO_FP32(x0_1->d))), y0->d);

        sumv1 = vmlaq_n_f32(sumv1, vaddq_f32(
                vmulq_n_f32(vcvtq_f32_s32(pl1), GGML_FP16_TO_FP32(x1_0->d)),
                vmulq_n_f32(vcvtq_f32_s32(ph1), GGML_FP16_TO_FP32(x1_1->d))), y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m128 d0 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 0].d));
        const __m128 d1 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 1].d));
        const __m256 d = _mm256_mul_ps(_mm256_set_m128(d1, d0), _mm256_broadcast_ss(&y[i].d));

        __m128i bx0 = bytes_from_nibbles_16(x[2*i + 0].qs);
        __m128i bx1 = bytes_from_nibbles_16(x[2*i + 1].qs);
        __m256i bx = _mm256_set_m128i(bx1, bx0);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8(8);
        bx = _mm256_sub_epi8(bx, off);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#else
    // scalar
    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[2*i + 0].qs;
        const uint8_t * restrict x1 = x[2*i + 1].qs;
        const  int8_t * restrict y0 = y[i].qs;

        const float d0 = GGML_FP16_TO_FP32(x[2*i + 0].d);
        const float d1 = GGML_FP16_TO_FP32(x[2*i + 1].d);

        int sumi_0 = 0;
        int sumi_1 = 0;

        for (int j = 0; j < QK8_0/4; j++) {
            const uint8_t v0 = x0[j];
            const uint8_t v1 = x1[j];

            const int i0_0 = (int8_t) (v0 & 0x0F) - 8;
            const int i1_0 = (int8_t) (v0 >>   4) - 8;

            const int i0_1 = (int8_t) (v1 & 0x0F) - 8;
            const int i1_1 = (int8_t) (v1 >>   4) - 8;

            const int i2_0 = y0[2*j + 0];
            const int i3_0 = y0[2*j + 1];

            const int i2_1 = y0[2*(j + QK8_0/4) + 0];
            const int i3_1 = y0[2*(j + QK8_0/4) + 1];

            sumi_0 += i0_0*i2_0 + i1_0*i3_0;
            sumi_1 += i0_1*i2_1 + i1_1*i3_1;
        }

        sumf += (d0 * y[i].d) * sumi_0;
        sumf += (d1 * y[i].d) * sumi_1;
    }
    *s = sumf;
#endif
}

static void ggml_vec_dot_q4_3_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_1;

    assert(n % QK8_1 == 0);
    assert(nb % 2 == 0);
    assert(QK8_1 == 2*QK4_3);

    const block_q4_3 * restrict x = vx;
    const block_q8_1_v2 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs0 = 0.0f;
    float summs1 = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const block_q4_3 * restrict x0_0 = &x[2*(i + 0) + 0];
        const block_q4_3 * restrict x0_1 = &x[2*(i + 0) + 1];

        const block_q8_1_v2 * restrict y0 = &y[i + 0];

        summs0 += GGML_FP16_TO_FP32(x0_0->m) * y0->s0;
        summs1 += GGML_FP16_TO_FP32(x0_1->m) * y0->s1;

        const uint8x16_t v0_0 = vcombine_u8(vld1_u8(x0_0->qs), vld1_u8(x0_1->qs));

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, vdupq_n_u8(0x0F)));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

        // interleave
        const int8x16_t v0_0lz = vzip1q_s8(v0_0l, v0_0h);
        const int8x16_t v0_0hz = vzip2q_s8(v0_0l, v0_0h);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);

        const float x0_0d = GGML_FP16_TO_FP32(x0_0->d);
        const float x0_1d = GGML_FP16_TO_FP32(x0_1->d);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0lz, v1_0l)), x0_0d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vdotq_s32(vdupq_n_s32(0), v0_0hz, v1_0h)), x0_1d*y0->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0lz), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0lz), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hz), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hz), vget_high_s8(v1_0h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(pl0), x0_0d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(ph0), x0_1d*y0->d);
#endif
    }

    *s = vaddvq_f32(vaddq_f32(sumv0, sumv1)) + summs0 + summs1;
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    // Main loop
    for (int i = 0; i < nb; i++) {
        const __m128 d0 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 0].d));
        const __m128 d1 = _mm_set1_ps(GGML_FP16_TO_FP32(x[2*i + 1].d));
        const __m256 dx = _mm256_set_m128(d1, d0);

        summs += GGML_FP16_TO_FP32(x[2*i + 0].m) * y[i].s0
               + GGML_FP16_TO_FP32(x[2*i + 1].m) * y[i].s1;

        const __m128i bx0 = bytes_from_nibbles_16(x[2*i + 0].qs);
        const __m128i bx1 = bytes_from_nibbles_16(x[2*i + 1].qs);
        const __m256i bx = _mm256_set_m128i(bx1, bx0);

        const __m256 dy = _mm256_broadcast_ss(&y[i].d);
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    *s = hsum_float_8(acc) + summs;
#else
    // scalar
    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[2*i + 0].qs;
        const uint8_t * restrict x1 = x[2*i + 1].qs;
        const  int8_t * restrict y0 = y[i].qs;

        const float d0 = GGML_FP16_TO_FP32(x[2*i + 0].d);
        const float m0 = GGML_FP16_TO_FP32(x[2*i + 0].m);
        const float d1 = GGML_FP16_TO_FP32(x[2*i + 1].d);
        const float m1 = GGML_FP16_TO_FP32(x[2*i + 1].m);

        int sxy_0 = 0;
        int sxy_1 = 0;

        for (int j = 0; j < QK8_1/4; j++) {
            const uint8_t v0 = x0[j];
            const uint8_t v1 = x1[j];

            const int x0_0 = v0 & 0x0F;
            const int x1_0 = v0 >> 4;

            const int x0_1 = v1 & 0x0F;
            const int x1_1 = v1 >> 4;

            const int y0_0 = y0[2*j + 0];
            const int y1_0 = y0[2*j + 1];

            const int y0_1 = y0[2*(j + QK8_1/4) + 0];
            const int y1_1 = y0[2*(j + QK8_1/4) + 1];

            sxy_0 += x0_0*y0_0 + x1_0*y1_0;
            sxy_1 += x0_1*y0_1 + x1_1*y1_1;
        }

        sumf += (d0*sxy_0 + d1*sxy_1)*y[i].d + m0*y[i].s0 + m1*y[i].s1;
    }
    *s = sumf;
#endif
}

static void ggml_vec_dot_q5_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);
    assert(QK8_0 == QK5_0);

    const block_q5_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);

    uint64_t tmp[4];

    for (int i = 0; i < nb; ++i) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q8_0 * restrict y0 = &y[i];

        const uint8x16_t m4b  = vdupq_n_u8(0x0F);
        const int8x16_t  s16b = vdupq_n_s8(0x10);

        // extract the 5th bit
        uint32_t qh;
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const int8x16_t qhl = vld1q_s8((const int8_t *)(tmp + 0));
        const int8x16_t qhh = vld1q_s8((const int8_t *)(tmp + 2));

        const uint8x16_t v0 = vld1q_u8(x0->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8  (v0, m4b));
        const int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));

        // interleave
        const int8x16_t v0lz = vzip1q_s8(v0l, v0h);
        const int8x16_t v0hz = vzip2q_s8(v0l, v0h);

        // add high bit and sub 16
        const int8x16_t v0lf = vsubq_s8(vorrq_s8(v0lz, qhl), s16b);
        const int8x16_t v0hf = vsubq_s8(vorrq_s8(v0hz, qhh), s16b);

        // load y
        const int8x16_t v1l = vld1q_s8(y0->qs);
        const int8x16_t v1h = vld1q_s8(y0->qs + 16);

        const float x0d = GGML_FP16_TO_FP32(x0->d);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0lf, v1l),
                        vdotq_s32(vdupq_n_s32(0), v0hf, v1h))), x0d*y0->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0lf), vget_low_s8 (v1l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0lf), vget_high_s8(v1l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0hf), vget_low_s8 (v1h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0hf), vget_high_s8(v1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));

        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0d*y0->d);
#endif
    }

    *s = vaddvq_f32(sumv);
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint64_t tmp[4];

    for (int i = 0; i < nb; ++i) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q8_0 * restrict y0 = &y[i];

        const v128_t m4b  = wasm_i8x16_splat(0x0F);
        const v128_t s16b = wasm_i8x16_splat(0x10);

        // extract the 5th bit
        uint32_t qh;
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // interleave
        const v128_t v0lz = wasm_v8x16_shuffle(v0l, v0h,  0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23);
        const v128_t v0hz = wasm_v8x16_shuffle(v0l, v0h,  8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

        // add high bit and sub 16
        const v128_t v0lf = wasm_i8x16_sub(wasm_v128_or(v0lz, qhl), s16b);
        const v128_t v0hf = wasm_i8x16_sub(wasm_v128_or(v0hz, qhh), s16b);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        const float x0d = GGML_FP16_TO_FP32(x0->d);

        // dot product
        sumv = wasm_f32x4_add(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))), wasm_f32x4_splat(x0d*y0->d)));
    }

    *s = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
         wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_mul_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d)), _mm256_broadcast_ss(&y[i].d));

        __m256i bx = bytes_from_nibbles_32_v2(x[i].qs);
        __m256i bxhi = bytes_from_bits_32(x[i].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        bx = _mm256_or_si256(bx, bxhi);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#else
    // scalar
    float sumf = 0.0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[i].qs;
        const  int8_t * restrict y0 = y[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        const float d = GGML_FP16_TO_FP32(x[i].d);

        int sxy = 0;

        for (int j = 0; j < QK8_0/2; j++) {
            const uint8_t v0 = x0[j];

            const int x0_0h = ((qh & (1u << (2*j + 0))) >> (2*j + 0)) << 4;
            const int x1_0h = ((qh & (1u << (2*j + 1))) >> (2*j + 1)) << 4;

            const int x0_0 = ((v0 & 0x0F) | x0_0h) - 16;
            const int x1_0 = ((v0 >>   4) | x1_0h) - 16;

            const int y0_0 = y0[2*j + 0];
            const int y1_0 = y0[2*j + 1];

            sxy += x0_0*y0_0 + x1_0*y1_0;
        }

        sumf += (d*sxy)*y[i].d;
    }
    *s = sumf;
#endif
}

static void ggml_vec_dot_q5_1_q8_1_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_1;

    assert(n % QK8_1 == 0);
    assert(nb % 2 == 0);
    assert(QK8_1 == QK5_1);

    const block_q5_1 * restrict x = vx;
    const block_q8_1_v2 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);

    float summs = 0.0f;

    uint64_t tmp[4];

    for (int i = 0; i < nb; ++i) {
        const block_q5_1 * restrict x0 = &x[i];
        const block_q8_1_v2 * restrict y0 = &y[i];

        summs += GGML_FP16_TO_FP32(x0->m) * (y0->s0 + y0->s1);

        // extract the 5th bit
        uint32_t qh;
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const int8x16_t qhl = vld1q_s8((const int8_t *)(tmp + 0));
        const int8x16_t qhh = vld1q_s8((const int8_t *)(tmp + 2));

        const uint8x16_t v0 = vld1q_u8(x0->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0l = vreinterpretq_s8_u8(vandq_u8  (v0, vdupq_n_u8(0x0F)));
        const int8x16_t v0h = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));

        // interleave
        const int8x16_t v0lz = vzip1q_s8(v0l, v0h);
        const int8x16_t v0hz = vzip2q_s8(v0l, v0h);

        // add
        const int8x16_t v0lf = vorrq_s8(v0lz, qhl);
        const int8x16_t v0hf = vorrq_s8(v0hz, qhh);

        // load y
        const int8x16_t v1l = vld1q_s8(y0->qs);
        const int8x16_t v1h = vld1q_s8(y0->qs + 16);

        const float x0d = GGML_FP16_TO_FP32(x0->d);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), v0lf, v1l),
                        vdotq_s32(vdupq_n_s32(0), v0hf, v1h))), x0d*y0->d);
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0lf), vget_low_s8 (v1l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0lf), vget_high_s8(v1l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0hf), vget_low_s8 (v1h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0hf), vget_high_s8(v1h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));

        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), x0d*y0->d);
#endif
    }

    *s = vaddvq_f32(sumv) + summs;
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    float summs = 0.0f;

    uint64_t tmp[4];

    for (int i = 0; i < nb; ++i) {
        const block_q5_1 * restrict x0 = &x[i];
        const block_q8_1_v2 * restrict y0 = &y[i];

        summs += GGML_FP16_TO_FP32(x0->m) * (y0->s0 + y0->s1);

        const v128_t m4b = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        uint32_t qh;
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        static bool x = true;

        // interleave
        const v128_t v0lz = wasm_v8x16_shuffle(v0l, v0h,  0, 16,  1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23);
        const v128_t v0hz = wasm_v8x16_shuffle(v0l, v0h,  8, 24,  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

        // add high bit
        const v128_t v0lf = wasm_v128_or(v0lz, qhl);
        const v128_t v0hf = wasm_v128_or(v0hz, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        const float x0d = GGML_FP16_TO_FP32(x0->d);

        // dot product
        sumv = wasm_f32x4_add(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))), wasm_f32x4_splat(x0d*y0->d)));
    }

    *s = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
         wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3) + summs;
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    // Main loop
    for (int i = 0; i < nb; i++) {
        const __m256 dx = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));

        summs += GGML_FP16_TO_FP32(x[i].m) * (y[i].s0 + y[i].s1);

        __m256i bx = bytes_from_nibbles_32_v2(x[i].qs);
        __m256i bxhi = bytes_from_bits_32(x[i].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        bx = _mm256_or_si256(bx, bxhi);

        const __m256 dy = _mm256_broadcast_ss(&y[i].d);
        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    *s = hsum_float_8(acc) + summs;
#else
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const uint8_t * restrict x0 = x[i].qs;
        const  int8_t * restrict y0 = y[i].qs;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);

        int sxy = 0;

        for (int j = 0; j < QK8_1/2; j++) {
            const uint8_t v0 = x0[j];

            const int x0_0h = ((qh & (1u << (2*j + 0))) >> (2*j + 0)) << 4;
            const int x1_0h = ((qh & (1u << (2*j + 1))) >> (2*j + 1)) << 4;

            const int x0_0 = (v0 & 0x0F) | x0_0h;
            const int x1_0 = (v0 >>   4) | x1_0h;

            const int y0_0 = y0[2*j + 0];
            const int y1_0 = y0[2*j + 1];

            sxy += x0_0*y0_0 + x1_0*y1_0;
        }

        sumf += (d*sxy)*y[i].d + m*(y[i].s0 + y[i].s1);
    }

    *s = sumf;
#endif
}

static void ggml_vec_dot_q8_0_q8_0_v2(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    const int nb = n / QK8_0;

    assert(n % QK8_0 == 0);
    assert(nb % 2 == 0);
    assert(QK8_0 == QK8_0);

    const block_q8_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i += 2) {
        const block_q8_0 * restrict x0 = &x[i + 0];
        const block_q8_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i + 0];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), x0->d*y0->d);

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), x1->d*y1->d);

#else
        const int16x8_t p0_0 = vmull_s8(vget_low_s8 (x0_0), vget_low_s8 (y0_0));
        const int16x8_t p0_1 = vmull_s8(vget_high_s8(x0_0), vget_high_s8(y0_0));
        const int16x8_t p0_2 = vmull_s8(vget_low_s8 (x0_1), vget_low_s8 (y0_1));
        const int16x8_t p0_3 = vmull_s8(vget_high_s8(x0_1), vget_high_s8(y0_1));

        const int16x8_t p1_0 = vmull_s8(vget_low_s8 (x1_0), vget_low_s8 (y1_0));
        const int16x8_t p1_1 = vmull_s8(vget_high_s8(x1_0), vget_high_s8(y1_0));
        const int16x8_t p1_2 = vmull_s8(vget_low_s8 (x1_1), vget_low_s8 (y1_1));
        const int16x8_t p1_3 = vmull_s8(vget_high_s8(x1_1), vget_high_s8(y1_1));

        const int32x4_t p0 = vaddq_s32(vpaddlq_s16(p0_0), vpaddlq_s16(p0_1));
        const int32x4_t p1 = vaddq_s32(vpaddlq_s16(p0_2), vpaddlq_s16(p0_3));
        const int32x4_t p2 = vaddq_s32(vpaddlq_s16(p1_0), vpaddlq_s16(p1_1));
        const int32x4_t p3 = vaddq_s32(vpaddlq_s16(p1_2), vpaddlq_s16(p1_3));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(p0, p1)), x0->d*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(p2, p3)), x1->d*y1->d);
#endif
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; ++i) {
        // Compute combined scale for the block
        const __m256 d = _mm256_mul_ps( _mm256_broadcast_ss( &x[i].d ), _mm256_broadcast_ss( &y[i].d ) );
        __m256i bx = _mm256_loadu_si256((const __m256i *)x[i].qs);
        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx, by);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    *s = hsum_float_8(acc);
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const int8_t * restrict x0 = x[i].qs;
        const int8_t * restrict y0 = y[i].qs;

        int sumi = 0;

        for (int j = 0; j < QK8_0; j++) {
            const int v0 = x0[j];
            const int v1 = y0[j];

            sumi += v0*v1;
        }

        sumf += (x[i].d*y[i].d)*sumi;
    }

    *s = sumf;
#endif
}



////////////////////////////////////////////////////////////////////////////////

size_t ggml_quantize_q4_0_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_0 == 0);
    const int nb = k / QK4_0;

    for (int j = 0; j < n; j += k) {
        block_q4_0 * restrict y = (block_q4_0 *)dst + j/QK4_0;

        quantize_row_q4_0_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_0; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_0*sizeof(block_q4_0));
}

size_t ggml_quantize_q4_1_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    for (int j = 0; j < n; j += k) {
        block_q4_1 * restrict y = (block_q4_1 *)dst + j/QK4_1;

        quantize_row_q4_1_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_1; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_1*sizeof(block_q4_1));
}

size_t ggml_quantize_q4_2_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_2 == 0);
    const int nb = k / QK4_2;

    for (int j = 0; j < n; j += k) {
        block_q4_2 * restrict y = (block_q4_2 *)dst + j/QK4_2;

        quantize_row_q4_2_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_2; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_2*sizeof(block_q4_2));
}

size_t ggml_quantize_q4_3_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK4_3 == 0);
    const int nb = k / QK4_3;

    for (int j = 0; j < n; j += k) {
        block_q4_3 * restrict y = (block_q4_3 *)dst + j/QK4_3;

        quantize_row_q4_3_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK4_3; l += 2) {
                const uint8_t vi0 = y[i].qs[l/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[l/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_3*sizeof(block_q4_3));
}

size_t ggml_quantize_q5_0_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK5_0 == 0);
    const int nb = k / QK5_0;

    for (int j = 0; j < n; j += k) {
        block_q5_0 * restrict y = (block_q5_0 *)dst + j/QK5_0;

        quantize_row_q5_0_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            uint32_t qh;
            memcpy(&qh, &y[i].qh, sizeof(qh));

            for (int l = 0; l < QK5_0; l += 2) {
                const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
                const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

                // cast to 16 bins
                const uint8_t vi0 = ((y[i].qs[l/2] & 0x0F) | vh0) / 2;
                const uint8_t vi1 = ((y[i].qs[l/2] >>   4) | vh1) / 2;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK5_0*sizeof(block_q5_0));
}

size_t ggml_quantize_q5_1_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK5_1 == 0);
    const int nb = k / QK5_1;

    for (int j = 0; j < n; j += k) {
        block_q5_1 * restrict y = (block_q5_1 *)dst + j/QK5_1;

        quantize_row_q5_1_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            uint32_t qh;
            memcpy(&qh, &y[i].qh, sizeof(qh));

            for (int l = 0; l < QK5_1; l += 2) {
                const uint8_t vh0 = ((qh & (1u << (l + 0))) >> (l + 0)) << 4;
                const uint8_t vh1 = ((qh & (1u << (l + 1))) >> (l + 1)) << 4;

                // cast to 16 bins
                const uint8_t vi0 = ((y[i].qs[l/2] & 0x0F) | vh0) / 2;
                const uint8_t vi1 = ((y[i].qs[l/2] >>   4) | vh1) / 2;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK5_1*sizeof(block_q5_1));
}

size_t ggml_quantize_q8_0_v2(const float * src, void * dst, int n, int k, int64_t * hist) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int j = 0; j < n; j += k) {
        block_q8_0 * restrict y = (block_q8_0 *)dst + j/QK8_0;

        quantize_row_q8_0_reference_v2(src + j, y, k);

        for (int i = 0; i < nb; i++) {
            for (int l = 0; l < QK8_0; ++l) {
                const int8_t vi = y[i].qs[l];

                hist[vi/16 + 8]++;
            }
        }
    }

    return (n/QK8_0*sizeof(block_q8_0));
}

//TODO: integrate
size_t ggml_quantize_chunk_v2(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist) {
    size_t result = 0;
    switch (type) {
        case GGML_TYPE_Q4_0:
            {
                GGML_ASSERT(start % QK4_0 == 0);
                block_q4_0 * block = (block_q4_0*)dst + start / QK4_0;
                result = ggml_quantize_q4_0_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_1:
            {
                GGML_ASSERT(start % QK4_1 == 0);
                block_q4_1 * block = (block_q4_1*)dst + start / QK4_1;
                result = ggml_quantize_q4_1_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_2:
            {
                GGML_ASSERT(start % QK4_2 == 0);
                block_q4_2 * block = (block_q4_2*)dst + start / QK4_2;
                result = ggml_quantize_q4_2_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q4_3:
            {
                GGML_ASSERT(start % QK4_3 == 0);
                block_q4_3 * block = (block_q4_3*)dst + start / QK4_3;
                result = ggml_quantize_q4_3_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q5_0:
            {
                GGML_ASSERT(start % QK5_0 == 0);
                block_q5_0 * block = (block_q5_0*)dst + start / QK5_0;
                result = ggml_quantize_q5_0_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q5_1:
            {
                GGML_ASSERT(start % QK5_1 == 0);
                block_q5_1 * block = (block_q5_1*)dst + start / QK5_1;
                result = ggml_quantize_q5_1_v2(src + start, block, n, n, hist);
            } break;
        case GGML_TYPE_Q8_0:
            {
                GGML_ASSERT(start % QK8_0 == 0);
                block_q8_0 * block = (block_q8_0*)dst + start / QK8_0;
                result = ggml_quantize_q8_0_v2(src + start, block, n, n, hist);
            } break;
        default:
            assert(false);
    }
    return result;
}

