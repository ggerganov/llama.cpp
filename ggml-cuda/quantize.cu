#include "quantize.cuh"

static __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

void quantize_row_q8_1_cuda(const float * x, void * vy, const int kx, const int ky, const int kx_padded, cudaStream_t stream) {
    const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ky, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

