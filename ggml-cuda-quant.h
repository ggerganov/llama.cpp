// quants kernels for ggml-cuda

// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 4
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 4
typedef struct {
    half    d;              // delta
    half    m;              // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 4
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 4
typedef struct {
    half d;                 // delta
    half m;                 // min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 8
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 8
typedef struct {
    half    d;              // delta
    half    s;              // unquantized sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

//================================= k-quants

#define QK_K 256

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half d;                  // super-block scale for quantized scales
    half dmin;               // super-block scale for quantized mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4]; // nibbles / quants
    uint8_t scales[3*QK_K/64];
    half d;
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + 11 * QK_K / 64, "wrong q3_K block size/padding");

typedef struct {
    half d;                    // super-block scale for quantized scales
    half dmin;                 // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");

typedef struct {
    half    d;                   // super-block scale for quantized scales
    half    dmin;                // super-block scale for quantized mins
    uint8_t scales[3*QK_K/64];   // scales, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");

typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");


template<typename src1_t, typename dst_t>
using dot_kernel_k_t = void (*)(const void * vx, const int ib, const int iqs, const src1_t * y, dst_t & v);

template<typename dst_t>
using vec_dot_q_cuda_t = dst_t (*)(const void * vbq, const block_q8_1 * bq8_1, const int iqs);


// TODO: f16
template<typename src_t>
static __global__ void quantize_q8_1(const src_t * x, void * vy, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i / QK8_0; // block index
    const int iqs = i % QK8_0; // quant index

    const float xi = x[i];
    float amax = fabsf(xi);
    float sum = xi;

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].d = d;
    y[ib].s = sum;
}

template<typename dst_t>
static __device__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dst_t d = x[ib].d;

    const uint8_t vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    const vec2_t<dst_t> off2 = make_vec2_t<dst_t>(8, 8);
    const vec2_t<dst_t> d2   = make_vec2_t<dst_t>(d, d);

    v = (v - off2) * d2;
}

template<typename dst_t>
static __device__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dst_t d = x[ib].d;
    const dst_t m = x[ib].m;

    const uint8_t vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    const vec2_t<dst_t> d2 = make_vec2_t<dst_t>(d, d);
    const vec2_t<dst_t> m2 = make_vec2_t<dst_t>(m, m);

    v = v * d2 + m2;
}

template<typename dst_t>
static __device__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dst_t d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    const vec2_t<dst_t> off2 = make_vec2_t<dst_t>(16, 16);
    const vec2_t<dst_t> d2   = make_vec2_t<dst_t>(d, d);

    v = (v - off2) * d2;
}

template<typename dst_t>
static __device__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dst_t d = x[ib].d;
    const dst_t m = x[ib].m;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const uint8_t xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    const vec2_t<dst_t> d2 = make_vec2_t<dst_t>(d, d);
    const vec2_t<dst_t> m2 = make_vec2_t<dst_t>(m, m);

    v = v * d2 + m2;
}

template<typename dst_t>
static __device__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, vec2_t<dst_t> & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dst_t d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    const vec2_t<dst_t> d2 = make_vec2_t<dst_t>(d, d);

    v = v * d2;
}

//================================== k-quants

static __global__ void dequantize_block_q2_K(const void * vx, float * yy) {

    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const block_q2_K * x = (const block_q2_K *) vx;

    const uint8_t q = x[i].qs[32*n + l];
    float * y = yy + i*QK_K + 128*n;

    float dall = x[i].d;
    float dmin = x[i].dmin;
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);

}

static __device__ void vec_dot_q2_K(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q2_K * x = (const block_q2_K *) vx;

    // if n is 0, we want to do the lower 128, else the upper 128,
    // covering y[l+0],  y[l+32], y[l+64], y[l+96] and
    //          y[l+16], y[l+48], y[l+80], y[l+112]
    int n = iqs/128;                // 0 or 1
    int r = iqs - 128*n;            // 0...120 in steps of 8
    int l = r/8;                    // 0...15 in steps of 1

    const float   * y = yy + 128*n + l;
    const uint8_t * q = x[ib].qs + 32*n + l;
    const uint8_t * s = x[ib].scales + 8*n;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    float sum = y[  0] * (dall * ((s[0] & 0xF) * ((q[ 0] >> 0) & 3)) - dmin * (s[0] >> 4))
              + y[ 32] * (dall * ((s[2] & 0xF) * ((q[ 0] >> 2) & 3)) - dmin * (s[2] >> 4))
              + y[ 64] * (dall * ((s[4] & 0xF) * ((q[ 0] >> 4) & 3)) - dmin * (s[4] >> 4))
              + y[ 96] * (dall * ((s[6] & 0xF) * ((q[ 0] >> 6) & 3)) - dmin * (s[6] >> 4))
              + y[ 16] * (dall * ((s[1] & 0xF) * ((q[16] >> 0) & 3)) - dmin * (s[1] >> 4))
              + y[ 48] * (dall * ((s[3] & 0xF) * ((q[16] >> 2) & 3)) - dmin * (s[3] >> 4))
              + y[ 80] * (dall * ((s[5] & 0xF) * ((q[16] >> 4) & 3)) - dmin * (s[5] >> 4))
              + y[112] * (dall * ((s[7] & 0xF) * ((q[16] >> 6) & 3)) - dmin * (s[7] >> 4));

    result = sum;

}

static __global__ void dequantize_block_q3_K(const void * vx, float * yy) {

    int r = threadIdx.x/4;
    int i = blockIdx.x;
    int tid = r/2;
    int is0 = r%2;
    int l0 = 16*is0 + 4*(threadIdx.x%4);
    int n = tid / 4;
    int j = tid - 4*n;

    const block_q3_K * x = (const block_q3_K *) vx;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    float * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));

}

static __device__ void vec_dot_q3_K(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q3_K * x = (const block_q3_K *) vx;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[3];
    uint32_t utmp[4];

    // if n is 0, we want to do the lower 128, else the upper 128,
    // covering y[l+0],  y[l+32], y[l+64], y[l+96] and
    //          y[l+16], y[l+48], y[l+80], y[l+112]
    int n = iqs/128;                // 0 or 1
    int r = iqs - 128*n;            // 0...120 in steps of 8
    int l = r/8;                    // 0...15 in steps of 1

    const float   * y = yy + 128*n + l;
    const uint8_t * q = x[ib].qs + 32*n + l;
    const uint8_t * hm = x[ib].hmask + l;
    const int8_t  * s = (const int8_t *)utmp + 8*n;

    memcpy(aux, x[ib].scales, 12);
    utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
    utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
    utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
    utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

    const float dall = x[ib].d;

    const uint8_t m = 1 << (4*n);

    float sum = y[  0] * (s[0] - 32) * (((q[ 0] >> 0) & 3) - (hm[ 0] & (m << 0) ? 0 : 4))
              + y[ 32] * (s[2] - 32) * (((q[ 0] >> 2) & 3) - (hm[ 0] & (m << 1) ? 0 : 4))
              + y[ 64] * (s[4] - 32) * (((q[ 0] >> 4) & 3) - (hm[ 0] & (m << 2) ? 0 : 4))
              + y[ 96] * (s[6] - 32) * (((q[ 0] >> 6) & 3) - (hm[ 0] & (m << 3) ? 0 : 4))
              + y[ 16] * (s[1] - 32) * (((q[16] >> 0) & 3) - (hm[16] & (m << 0) ? 0 : 4))
              + y[ 48] * (s[3] - 32) * (((q[16] >> 2) & 3) - (hm[16] & (m << 1) ? 0 : 4))
              + y[ 80] * (s[5] - 32) * (((q[16] >> 4) & 3) - (hm[16] & (m << 2) ? 0 : 4))
              + y[112] * (s[7] - 32) * (((q[16] >> 6) & 3) - (hm[16] & (m << 3) ? 0 : 4));

    result = sum * dall;

}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __global__ void dequantize_block_q4_K(const void * vx, float * yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = blockIdx.x;

    //// assume 64 threads - this is very slightly better than the one below
    //const int tid = threadIdx.x;
    //const int il  = tid/16;
    //const int ir  = tid%16;
    //const int is  = 2*il;
    //const int n   = 2;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    float * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = x[i].d;
    const float dmin = x[i].dmin;

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

static __device__ void vec_dot_q4_K(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q4_K * x = (const block_q4_K *) vx;

                                    // iqs is in 0...248 in steps of 8 =>
    const int j  = iqs / 64;        // j  is in 0...3
    const int ir = (iqs - 64*j)/2;  // ir is in 0...28 in steps of 4
    const int is = 2*j;             // is is in 0...6 in steps of 2

    const float   * y = yy + 64*j + ir;
    const uint8_t * q = x[ib].qs + 32*j + ir;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    float sum = 0;
    for (int k = 0; k < 4; ++k) {
        sum += y[k +  0] * (d1 * (q[k] & 0xF) - m1);
        sum += y[k + 32] * (d2 * (q[k] >>  4) - m2);
    }
    result = sum;

}

static __global__ void dequantize_block_q5_K(const void * vx, float * yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    float * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = x[i].d;
    const float dmin = x[i].dmin;

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

static __device__ void vec_dot_q5_K(const void * vx, const int ib, const int iqs, const float * yy, float & result) {

    const block_q5_K * x = (const block_q5_K *) vx;

                                    // iqs is in 0...248 in steps of 8 =>
    const int j  = iqs / 64;        // j  is in 0...3
    const int ir = (iqs - 64*j)/2;  // ir is in 0...28 in steps of 4
    const int is = 2*j;             // is is in 0...6 in steps of 2

    const float   * y  = yy + 64*j + ir;
    const uint8_t * ql = x[ib].qs + 32*j + ir;
    const uint8_t * qh = x[ib].qh + ir;

    const float dall = x[ib].d;
    const float dmin = x[ib].dmin;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    uint8_t   hm  = 1 << is;
    float sum = 0;
    for (int k = 0; k < 4; ++k) {
        sum += y[k +  0] * (d1 * ((ql[k] & 0xF) + (qh[k] & hm ? 16 : 0)) - m1);
    }
    hm <<= 1;
    for (int k = 0; k < 4; ++k) {
        sum += y[k + 32] * (d2 * ((ql[k] >>  4) + (qh[k] & hm ? 16 : 0)) - m2);
    }
    result = sum;

}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * vx, dst_t * yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int ip  = tid/32;   // ip is 0 or 1
    const int il  = tid - 32*ip; // 0...32
    const int is  = 8*ip + il/16;

    // TODO: fp16 compute
    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename src1_t, typename dst_t>
static __global__ void dequantize_mul_mat_vec_q6_k(const void * vx, const src1_t * yy, dst_t * dst, const int ncols, int nrows) {
    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    dst_t tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const src1_t  * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const dst_t d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        dst_t sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += (dst_t)y[l+ 0] * (dst_t)s[0] * d * (dst_t)((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + (dst_t)y[l+32] * (dst_t)s[2] * d * (dst_t)((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + (dst_t)y[l+64] * (dst_t)s[4] * d * (dst_t)((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + (dst_t)y[l+96] * (dst_t)s[6] * d * (dst_t)((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

template <typename dst_t, int qk, int qr, dequantize_kernel_t<dst_t> dequantize_kernel>
static __global__ void dequantize_block(const void * vx, dst_t * y, const int k) {
    const int i = blockDim.x*blockIdx.x + 2*threadIdx.x;

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    vec2_t<dst_t> v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template<typename dst_t>
static __device__ __forceinline__ dst_t vec_dot_q4_0_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int iqs) {
#if __CUDA_ARCH__ >= 600 // lowest compute capability for integer intrinsics
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int vi;
    memcpy(&vi,  &bq4_0->qs[sizeof(int) * (iqs + 0)], sizeof(int));
    const int ui0 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + 0)]);
    const int ui1 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + QI4_0)]);

    const float d = __half2float(bq4_0->d) * __half2float(bq8_1->d);

    // subtract 8 from each quantized value
    const int vi0 = __vsub4((vi >> 0) & 0x0F0F0F0F, 0x08080808);
    const int vi1 = __vsub4((vi >> 4) & 0x0F0F0F0F, 0x08080808);

    // SIMD dot product of quantized values
    int sumi = __dp4a(vi0, ui0, 0);
    sumi     = __dp4a(vi1, ui1, sumi);

    return sumi*d;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= 600
}

template<typename dst_t>
static __device__ __forceinline__ dst_t vec_dot_q4_1_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int iqs) {
#if __CUDA_ARCH__ >= 600 // lowest compute capability for integer intrinsics
    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    const int vi  = *((int *) &bq4_1->qs[sizeof(int) * (iqs + 0)]);
    const int ui0 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + 0)]);
    const int ui1 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + QI4_1)]);

    const float d = __half2float(bq4_1->d) * __half2float(bq8_1->d);
    const float m = bq4_1->m;
    const float s = bq8_1->s;

    const int vi0 = (vi >> 0) & 0x0F0F0F0F;
    const int vi1 = (vi >> 4) & 0x0F0F0F0F;

    // SIMD dot product of quantized values
    int sumi = __dp4a(vi0, ui0, 0);
    sumi     = __dp4a(vi1, ui1, sumi);

    return sumi*d + m*s / QI4_1; // scale sum by QI4_1 because there are QI4_1 threads working on this block
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= 600
}

template<typename dst_t>
static __device__ __forceinline__ dst_t vec_dot_q5_0_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int iqs) {
#if __CUDA_ARCH__ >= 600 // lowest compute capability for integer intrinsics
    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int qs;
    memcpy(&qs, &bq5_0->qs[sizeof(int) * (iqs + 0)], sizeof(int));
    const int qh0 = bq5_0->qh[iqs/2 + 0] >> 4*(iqs%2);
    const int qh1 = bq5_0->qh[iqs/2 + 2] >> 4*(iqs%2);
    const int ui0 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + 0)]);
    const int ui1 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + QI5_0)]);

    const float d = __half2float(bq5_0->d) * __half2float(bq8_1->d);

    int vi0 = (qs  >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh0 as 5th bits
    vi0    |= (qh0 <<  4) & 0x00000010; // 1 ->  5
    vi0    |= (qh0 << 11) & 0x00001000; // 2 -> 13
    vi0    |= (qh0 << 18) & 0x00100000; // 3 -> 21
    vi0    |= (qh0 << 25) & 0x10000000; // 4 -> 29
    vi0     = __vsub4(vi0,  0x10101010); // subtract 16 from quantized values
    int sumi = __dp4a(vi0, ui0, 0); // SIMD dot product of quantized values

    int vi1 = (qs  >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh1 as 5th bits
    vi1    |= (qh1 <<  4) & 0x00000010; // 1 ->  5
    vi1    |= (qh1 << 11) & 0x00001000; // 2 -> 13
    vi1    |= (qh1 << 18) & 0x00100000; // 3 -> 21
    vi1    |= (qh1 << 25) & 0x10000000; // 4 -> 29
    vi1     = __vsub4(vi1,  0x10101010); // subtract 16 from quantized values
    sumi = __dp4a(vi1, ui1, sumi); // SIMD dot product of quantized values

    return sumi*d;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= 600
}

template<typename dst_t>
static __device__ __forceinline__ dst_t vec_dot_q5_1_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int iqs) {
#if __CUDA_ARCH__ >= 600 // lowest compute capability for integer intrinsics
    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    const int qs  = *((int *) &bq5_1->qs[sizeof(int) * (iqs + 0)]);
    const int qh0 = bq5_1->qh[iqs/2 + 0] >> 4*(iqs%2);
    const int qh1 = bq5_1->qh[iqs/2 + 2] >> 4*(iqs%2);
    const int ui0 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + 0)]);
    const int ui1 = *((int *) &bq8_1->qs[sizeof(int) * (iqs + QI5_1)]);

    const float d = __half2float(bq5_1->d) * __half2float(bq8_1->d);
    const float m = bq5_1->m;
    const float s = bq8_1->s;

    int vi0 = (qs  >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh0 as 5th bits
    vi0    |= (qh0 <<  4) & 0x00000010; // 1 ->  5
    vi0    |= (qh0 << 11) & 0x00001000; // 2 -> 13
    vi0    |= (qh0 << 18) & 0x00100000; // 3 -> 21
    vi0    |= (qh0 << 25) & 0x10000000; // 4 -> 29
    int sumi = __dp4a(vi0, ui0, 0); // SIMD dot product of quantized values

    int vi1 = (qs  >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh1 as 5th bits
    vi1    |= (qh1 <<  4) & 0x00000010; // 1 ->  5
    vi1    |= (qh1 << 11) & 0x00001000; // 2 -> 13
    vi1    |= (qh1 << 18) & 0x00100000; // 3 -> 21
    vi1    |= (qh1 << 25) & 0x10000000; // 4 -> 29
    sumi = __dp4a(vi1, ui1, sumi); // SIMD dot product of quantized values

    return sumi*d + m*s / QI5_1; // scale sum by QI5_1 because there are QI5_1 threads working on this block
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= 600
}

template<typename dst_t>
static __device__ __forceinline__ dst_t vec_dot_q8_0_q8_1(const void * vbq, const block_q8_1 * bq8_1, const int iqs) {
#if __CUDA_ARCH__ >= 600 // lowest compute capability for integer intrinsics
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int vi;
    memcpy(&vi,  &bq8_0->qs[sizeof(int) * (iqs + 0)], sizeof(int));
    const int ui = *((int *) &bq8_1->qs[sizeof(int) * (iqs + 0)]);

    const float d = __half2float(bq8_0->d) * __half2float(bq8_1->d);

    // SIMD dot product of quantized values
    int sumi = __dp4a(vi, ui, 0);

    return sumi*d;
#else
    return 0.0f; // only to satisfy the compiler
#endif // __CUDA_ARCH__ >= 600
}

template <typename dst_t, int qk, int qi, typename block_q_t, vec_dot_q_cuda_t<dst_t> vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(const void * vx, const void * vy, dst_t * dst, const int ncols, const int nrows) {
    const int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = WARP_SIZE / qi;

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = 0; i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i + threadIdx.x / qi; // x block index

        const int iby = i + threadIdx.x / qi; // y block index

        const int iqs  = threadIdx.x % qi; // x block quant index when casting the quants to int

        tmp += (float)vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = (dst_t)tmp;
    }
}

template <typename src1_t, typename dst_t, int qk, int qr, dequantize_kernel_t<dst_t> dequantize_kernel>
static __global__ void dequantize_mul_mat_vec(const void * vx, const src1_t * y, dst_t * dst, const int ncols, const int nrows) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

    vec2_t<dst_t> tmp2 = make_vec2_t<dst_t>(0, 0); // partial sum for thread in warp

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val

            // dequantize
            vec2_t<dst_t> xc;
            dequantize_kernel(vx, ib, iqs + j/qr, xc);

            // matrix multiplication
            vec2_t<dst_t> yc = make_vec2_t<dst_t>(
                y[iybs + iqs + j/qr + 0],
                y[iybs + iqs + j/qr + y_offset]);
            tmp2 += xc * yc;
        }
    }

    // sum up partial sums and write back result
    // TODO: reducing as half2 may be faster, but requires special handling for float2
    dst_t tmp = tmp2.x + tmp2.y;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

template <typename src1_t, typename dst_t, int n_thread, dot_kernel_k_t<src1_t, dst_t> dot_kernel>
static __global__ void dequantize_mul_mat_vec_k(const void * vx, const src1_t * y, dst_t * dst, const int ncols) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int iter_stride = QK_K;
    const int vals_per_iter = iter_stride / n_thread;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    dst_t tmp = 0; // partial sum for thread in warp

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = ib0 + col/QK_K; // x block index
        const int iqs = col%QK_K; // x quant index
        const int iybs = col - col%QK_K; // y block start index

        dst_t v;
        dot_kernel(vx, ib, iqs, y + iybs, v);
        tmp += v;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}
