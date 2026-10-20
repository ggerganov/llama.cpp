#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK_K 256
#define Q2_K_MV_R 1
#define Q2_K_MV_NSG 1

inline float4 q2_k_vals(uint pk) {
    return (float4)((float)( pk       & 3u), (float)((pk >> 2) & 3u),
                    (float)((pk >> 4) & 3u), (float)((pk >> 6) & 3u));
}

kernel void kernel_gemv_noshuffle_q2_k_f32(
        global const uchar * src0_qs,
        global const uchar * src0_sc,
        global const half  * src0_dm,
        global const float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,      // K
        int ne01,      // M
        int ne10,      // activation row stride, == K
        int ne0        // dst row stride
) {
    src1 = (global const float *)((global const char *)src1 + offset1);
    dst  = (global float       *)((global char       *)dst  + offsetd);

    const uint m   = (uint)ne01;
    const uint K   = (uint)ne00;
    const uint nsb = K / QK_K;

    const uint lid = get_local_id(0);
    const uint sgi = get_local_id(1);
    const uint col = get_group_id(1);

    global const float * y = src1 + (ulong)col * (uint)ne10;

    const uint mr  = m / Q2_K_MV_R;                  // row groups
    const uint j   = get_group_id(0) * 64u + lid;   // row group this lane owns
    const uint row = j * Q2_K_MV_R;

    float sumf0 = 0.f;

    if (j < mr) {
        for (uint ib = sgi; ib < nsb; ib += Q2_K_MV_NSG) {
            const half2 dm = vload2(row + ib * m, src0_dm);

            float ad = 0.f, am = 0.f;
            for (uint sb = 0; sb < 8u; ++sb) {
                const uint grp = ib * 64u + sb * 8u;
                const uint qsb = row + grp * m;

                for (uint h = 0; h < 2u; ++h) {
                    const uint sc = (uint)src0_sc[row + (2u * (ib * 8u + sb) + h) * m];

                    float a = 0.f;
                    float4 as = (float4)(0.f);
                    for (uint u = 0; u < 4u; ++u) {
                        const uint gg = 4u*h + u;
                        const float4 yv = vload4(grp + gg, y);
                        as += yv;
                        const uint pkv = (uint)src0_qs[qsb + gg * m];
                        a += dot(yv, q2_k_vals(pkv));
                    }
                    const float asum = as.s0 + as.s1 + as.s2 + as.s3;
                    ad += (float)(sc & 0xFu) * a;   am += (float)(sc >> 4) * asum;
                }
            }
            sumf0 += (float)dm.s0 * ad - (float)dm.s1 * am;
        }
    }
    if (j < mr) {
        dst[(ulong)col * (uint)ne0 + row] = sumf0;
    }
}
