uint top_k_key(float value) {
    uint bits = as_uint(value);
    if ((bits & 0x7fffffffu) == 0) {
        bits = 0;
    }
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

kernel void kernel_top_k_f32(
        global const char * src,
        ulong offset0,
        global int * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int k,
        local int * counts) {
    const ulong row = get_group_id(1);
    const int tid = get_local_id(0);
    const int nth = get_local_size(0);
    src += offset0 + (row % ne01)*nb01 + (row / ne01 % ne02)*nb02 + (row / ne01 / ne02)*nb03;
    dst = (global int *) ((global char *) dst + offsetd) + row*k;

    if (k == ne00) {
        for (int i = tid; i < ne00; i += nth) {
            dst[i] = i;
        }
        return;
    }

    // Find the kth largest float key without sorting or allocating a row-sized buffer.
    uint prefix = 0;
    uint mask = 0;
    int rank = k;
    for (int bit = 31; bit >= 0; --bit) {
        const uint candidate = prefix | (1u << bit);
        mask |= 1u << bit;
        int count = 0;
        for (int i = tid; i < ne00; i += nth) {
            const uint key = top_k_key(*(global const float *) (src + (ulong) i*nb00));
            count += (key & mask) == candidate;
        }
        counts[tid] = count;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = nth/2; stride > 0; stride /= 2) {
            if (tid < stride) {
                counts[tid] += counts[tid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const int total = counts[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (total >= rank) {
            prefix = candidate;
        } else {
            rank -= total;
        }
    }

    int greater = 0;
    int equal = 0;
    for (int i = tid; i < ne00; i += nth) {
        const uint key = top_k_key(*(global const float *) (src + (ulong) i*nb00));
        greater += key > prefix;
        equal += key == prefix;
    }
    counts[tid] = greater;
    counts[nth + tid] = equal;
    barrier(CLK_LOCAL_MEM_FENCE);

    int pos_greater = 0;
    int pos_equal = 0;
    for (int i = 0; i < nth; ++i) {
        pos_equal += counts[i];
        if (i < tid) {
            pos_greater += counts[i];
            pos_equal += counts[nth + i];
        }
    }
    for (int i = tid; i < ne00; i += nth) {
        const uint key = top_k_key(*(global const float *) (src + (ulong) i*nb00));
        if (key > prefix) {
            dst[pos_greater++] = i;
        } else if (key == prefix && pos_equal < k) {
            dst[pos_equal++] = i;
        }
    }
}
