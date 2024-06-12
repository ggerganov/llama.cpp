#include "upscale.cuh"

static __global__ void upscale_f32(const float * x, float * dst, const int ne00, const int ne00xne01, const int scale_factor) {
    // blockIdx.z: idx of ne02*ne03
    // blockIdx.y: idx of ne01*scale_factor， aka ne1
    // blockIDx.x: idx of ne00*scale_factor / BLOCK_SIZE
    // ne00xne01: ne00 * ne01
    int ne0 = ne00 * scale_factor;
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    // operation
    int i00 = nidx / scale_factor;
    int i01 = blockIdx.y / scale_factor;
    int offset_src =
        i00 +
        i01 * ne00 +
        blockIdx.z * ne00xne01;
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    dst[offset_dst] = x[offset_src];
}

static void upscale_f32_cuda(const float * x, float * dst, const int ne00, const int ne01, const int ne02, const int ne03,
                             const int scale_factor, cudaStream_t stream) {
    int ne0 = (ne00 * scale_factor);
    int num_blocks = (ne0 + CUDA_UPSCALE_BLOCK_SIZE - 1) / CUDA_UPSCALE_BLOCK_SIZE;
    dim3 gridDim(num_blocks, (ne01 * scale_factor), ne02*ne03);
    upscale_f32<<<gridDim, CUDA_UPSCALE_BLOCK_SIZE, 0, stream>>>(x, dst, ne00, ne00 * ne01, scale_factor);
}

void ggml_cuda_op_upscale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

    const int scale_factor = dst->op_params[0];

    upscale_f32_cuda(src0_d, dst_d, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], scale_factor, stream);
}
