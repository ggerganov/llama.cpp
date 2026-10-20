#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_conv2d_dw_f32_f32(
        global const float * weights,
        ulong                weights_offset,
        global const float * input,
        ulong                input_offset,
        global       float * output,
        ulong                output_offset,
        ulong C, ulong N,
        ulong KW, ulong KH, ulong W, ulong H, ulong OW, ulong OH,
        uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
        ulong nb01, ulong nb03,
        ulong nb11, ulong nb12, ulong nb13,
        ulong nb1, ulong nb2, ulong nb3
) {
    weights = (global const float *) ((global const char *) weights + weights_offset);
    input = (global const float *) ((global const char *) input + input_offset);
    output = (global float *) ((global char *) output + output_offset);

    const ulong gid = get_global_id(0);
    const ulong OHW = OH * OW;
    const ulong COHW = C * OHW;
    const ulong total = N * COHW;
    if (gid >= total) {
        return;
    }

    const ulong n = gid / COHW;
    const ulong c_rem = gid - n * COHW;
    const ulong c = c_rem / OHW;
    const ulong oh = (c_rem - c * OHW) / OW;
    const ulong ow = c_rem - c * OHW - oh * OW;

    float sum = 0.0f;
    for (ulong kh = 0; kh < KH; ++kh) {
        const long ih = (long) oh * s1 + (long) kh * d1 - p1;
        if (ih < 0 || ih >= (long) H) {
            continue;
        }
        for (ulong kw = 0; kw < KW; ++kw) {
            const long iw = (long) ow * s0 + (long) kw * d0 - p0;
            if (iw < 0 || iw >= (long) W) {
                continue;
            }
            const ulong weight_idx = kw + kh * nb01 + c * nb03;
            const ulong input_idx = (ulong) iw + (ulong) ih * nb11 + c * nb12 + n * nb13;
            sum += weights[weight_idx] * input[input_idx];
        }
    }

    const ulong output_idx = ow + oh * nb1 + c * nb2 + n * nb3;
    output[output_idx] = sum;
}

kernel void kernel_conv2d_dwf16_f32(
        global const half * weights,
        ulong               weights_offset,
        global const float * input,
        ulong                input_offset,
        global       float * output,
        ulong                output_offset,
        ulong C, ulong N,
        ulong KW, ulong KH, ulong W, ulong H, ulong OW, ulong OH,
        uint s0, uint s1, uint p0, uint p1, uint d0, uint d1,
        ulong nb01, ulong nb03,
        ulong nb11, ulong nb12, ulong nb13,
        ulong nb1, ulong nb2, ulong nb3
) {
    weights = (global const half *) ((global const char *) weights + weights_offset);
    input = (global const float *) ((global const char *) input + input_offset);
    output = (global float *) ((global char *) output + output_offset);

    const ulong gid = get_global_id(0);
    const ulong OHW = OH * OW;
    const ulong COHW = C * OHW;
    const ulong total = N * COHW;
    if (gid >= total) {
        return;
    }

    const ulong n = gid / COHW;
    const ulong c_rem = gid - n * COHW;
    const ulong c = c_rem / OHW;
    const ulong oh = (c_rem - c * OHW) / OW;
    const ulong ow = c_rem - c * OHW - oh * OW;

    float sum = 0.0f;
    for (ulong kh = 0; kh < KH; ++kh) {
        const long ih = (long) oh * s1 + (long) kh * d1 - p1;
        if (ih < 0 || ih >= (long) H) {
            continue;
        }
        for (ulong kw = 0; kw < KW; ++kw) {
            const long iw = (long) ow * s0 + (long) kw * d0 - p0;
            if (iw < 0 || iw >= (long) W) {
                continue;
            }
            const ulong weight_idx = kw + kh * nb01 + c * nb03;
            const ulong input_idx = (ulong) iw + (ulong) ih * nb11 + c * nb12 + n * nb13;
            sum += (float) weights[weight_idx] * input[input_idx];
        }
    }

    const ulong output_idx = ow + oh * nb1 + c * nb2 + n * nb3;
    output[output_idx] = sum;
}
