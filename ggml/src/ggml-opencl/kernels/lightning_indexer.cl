#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_lightning_indexer(
        global const char * q,
        ulong offset_q,
        global const char * k,
        ulong offset_k,
        global const char * w,
        ulong offset_w,
        global const char * mask,
        ulong offset_m,
        global float * dst,
        ulong offset_d,
        int head_size,
        int n_heads,
        int n_tokens,
        int n_keys,
        int n_masks,
        int type_k,
        ulong nbq1,
        ulong nbq2,
        ulong nbq3,
        ulong nbk2,
        ulong nbk3,
        ulong nbw1,
        ulong nbw3,
        ulong nbm1,
        ulong nbm3) {
    const int tid = get_local_id(0);
    const int lane = tid % 16;
    const int key = get_group_id(0)*4 + tid/16;
    const int token = get_group_id(1);
    const int stream = get_group_id(2);
    local float sums[64];

    q += offset_q + token*nbq2 + stream*nbq3;
    k += offset_k + (ulong) key*nbk2 + stream*nbk3;
    w += offset_w + token*nbw1 + stream*nbw3;
    mask += offset_m + token*nbm1 + (stream % n_masks)*nbm3;
    dst = (global float *) ((global char *) dst + offset_d);

    float score = 0.0f;
    for (int h = 0; h < n_heads; ++h) {
        global const float * q_head = (global const float *) (q + h*nbq1);
        float sum = 0.0f;
        if (key < n_keys) {
            for (int i = lane; i < head_size; i += 16) {
                float kv;
                if (type_k == 0) {
                    kv = ((global const float *) k)[i];
                } else {
                    kv = (float) ((global const half *) k)[i];
                }
                sum = mad(q_head[i], kv, sum);
            }
        }
        sums[tid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 8; stride > 0; stride /= 2) {
            if (lane < stride) {
                sums[tid] += sums[tid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0) {
            score = mad(fmax(sums[tid], 0.0f), ((global const float *) w)[h], score);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0 && key < n_keys) {
        dst[((ulong) stream*n_tokens + token)*n_keys + key] = score + (float) ((global const half *) mask)[key];
    }
}
