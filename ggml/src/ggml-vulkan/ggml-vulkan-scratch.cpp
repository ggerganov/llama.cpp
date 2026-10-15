#include "ggml-vulkan-common.h"

size_t ggml_vk_scratch_offset(const vk_device & device, size_t nbytes) {
    return ggml_vk_align_size(nbytes, device->properties.limits.minStorageBufferOffsetAlignment);
}

size_t ggml_vk_op_scratch_size(const vk_device & device, const ggml_tensor * dst) {
    size_t scratch = 0;
    switch (dst->op) {
        default:
            break;
    }
    if (scratch != 0 &&
        ggml_vk_scratch_offset(device, ggml_nbytes(dst)) + scratch > device->suballocation_block_size) {
        return 0;
    }
    return scratch;
}

// Total allocation span of a node output, including its dst scratch padding.
// Equals ggml_nbytes(dst) when the op has no scratch or dst is a view.
size_t ggml_vk_tensor_alloc_size(const vk_device & device, const ggml_tensor * dst) {
    const size_t nbytes = ggml_nbytes(dst);
    if (dst->view_src != nullptr) {
        return nbytes;
    }
    const size_t scratch = ggml_vk_op_scratch_size(device, dst);
    if (scratch == 0) {
        return nbytes;
    }
    return ggml_vk_scratch_offset(device, nbytes) + scratch;
}

bool ggml_vk_scratch_available(const ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    if (dst->view_src != nullptr) {
        return false;
    }
    if (dst->buffer == nullptr || !ggml_backend_buffer_is_vk(dst->buffer)) {
        return false;
    }
    return ggml_vk_op_scratch_size(ctx->device, dst) > 0;
}

vk_subbuffer ggml_vk_scratch_subbuffer(const ggml_backend_vk_context * ctx, const ggml_tensor * dst, size_t offset, size_t size) {
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    const size_t base = vk_tensor_offset(dst) + ggml_vk_scratch_offset(ctx->device, ggml_nbytes(dst)) + offset;
    return { buf_ctx->dev_buffer, base, size };
}
