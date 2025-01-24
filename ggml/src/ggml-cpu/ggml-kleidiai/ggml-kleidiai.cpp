// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#include <arm_neon.h>
#include <assert.h>
#include <cfloat>
#include <stdint.h>
#include <string.h>
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <string_view>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#include <excpt.h>
#endif

#include "ggml-kleidiai.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "kleidiai_kernels.h"

#include "kai_common.h"

static const size_t k_q4_0_block_size = 32;

struct ggml_kleidiai_context {
    ggml_kleidiai_kernels * kernels;
} static ctx = { NULL };

static void init_kleidiai_context(int n_threads) {
    static bool initialized = false;

    if (!initialized) {
        GGML_ASSERT(n_threads > 0);

        initialized = true;

        cpu_feature features  = (ggml_cpu_has_dotprod()     ? CPU_FEATURE_DOTPROD : CPU_FEATURE_NONE) |
                                (ggml_cpu_has_matmul_int8() ? CPU_FEATURE_I8MM    : CPU_FEATURE_NONE) |
                                (ggml_cpu_has_sve()         ? CPU_FEATURE_SVE     : CPU_FEATURE_NONE);

#if defined(__APPLE__)
        if (n_threads == 1) {
            features |= ggml_cpu_has_sme() ? CPU_FEATURE_SME : CPU_FEATURE_NONE;
        }
#else
        features |= ggml_cpu_has_sme() ? CPU_FEATURE_SME : CPU_FEATURE_NONE;
#endif
        ctx.kernels = ggml_kleidiai_select_kernels(features);
    }
}

namespace ggml::cpu::kleidiai {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        GGML_ASSERT(ctx.kernels);
        kernel_info * kernel = op->src[1]->ne[1] == 1 ? &ctx.kernels->gemv : &ctx.kernels->gemm;

        size_t k = op->src[0]->ne[0];
        size_t m = op->src[1]->ne[1];

        size_t mr = kernel->get_mr();
        size_t kr = kernel->get_kr();
        size_t sr = kernel->get_sr();
        size_t bl = k_q4_0_block_size;

        size = ctx.kernels->lhs_info.packed_size(m, k, bl, mr, kr, sr);

        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override {
        if (dst->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * src0 = dst->src[0];
            const ggml_tensor * src1 = dst->src[1];

            GGML_TENSOR_BINARY_OP_LOCALS

            GGML_ASSERT(ctx.kernels);
            kernel_info * kernel = src1->ne[1] == 1 ? &ctx.kernels->gemv : &ctx.kernels->gemm;
            lhs_packing_info * lhs_info = &ctx.kernels->lhs_info;

            GGML_ASSERT(kernel);

            const int ith = params->ith;
            const int nth = params->nth;

            const size_t k = ne00;
            const size_t m = ne11;
            const size_t n = ne01;

            const size_t n_step = kernel->get_n_step();
            const size_t num_n_per_thread = kai_roundup(kai_roundup(n, nth) / nth, n_step);
            const size_t n_start = ith * num_n_per_thread;

            size_t n_to_process = num_n_per_thread;
            if ((n_start + n_to_process) > n) {
                n_to_process = n - n_start;
            }

            const uint8_t * lhs        = static_cast<const uint8_t *>(src1->data);
            uint8_t * lhs_packed       = (uint8_t*)params->wdata;
            const uint8_t * rhs_packed = static_cast<const uint8_t *>(src0->data);

            size_t mr = kernel->get_mr();
            size_t kr = kernel->get_kr();
            size_t sr = kernel->get_sr();
            size_t bl = k_q4_0_block_size;

            const size_t lhs_packed_offset = lhs_info->get_packed_offset(0, k, bl, mr, kr, sr);

            if (ith == 0) {
                // Transform LHS
                const size_t src_stride = src1->nb[1];
                const float * src_ptr = reinterpret_cast<const float *>(lhs + lhs_info->get_offset(0, dst->src[1]->nb[1]));
                void * dst_ptr = static_cast<void *>(lhs_packed + lhs_packed_offset);

                lhs_info->pack_func(m, k, bl, mr, kr, sr, 0, src_ptr, src_stride, dst_ptr);
            }

            ggml_barrier(params->threadpool);
            // Perform the operation
            const size_t dst_stride = dst->nb[1];

            const size_t rhs_packed_offset = kernel->get_rhs_packed_offset(n_start, k, k_q4_0_block_size);
            const size_t dst_offset        = kernel->get_dst_offset(0, n_start, dst_stride);

            const void * lhs_ptr = static_cast<const void *>(lhs_packed + lhs_packed_offset);
            const void * rhs_ptr = static_cast<const void *>(rhs_packed + rhs_packed_offset);
            float *dst_ptr = reinterpret_cast<float *>(static_cast<uint8_t *>(dst->data) + dst_offset);

            kernel->run_kernel(m, n_to_process, k, k_q4_0_block_size, lhs_ptr, rhs_ptr, dst_ptr,
                                 dst_stride, sizeof(float), -FLT_MAX, FLT_MAX);
            return true;
        }
        return false;
    }

public:
    int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) {
        GGML_ASSERT(ctx.kernels);
        const size_t n = tensor->ne[1];
        const size_t k = tensor->ne[0];
        size_t nr      = ctx.kernels->gemm.get_nr();
        size_t kr      = ctx.kernels->gemm.get_kr();
        size_t sr      = ctx.kernels->gemm.get_sr();

#ifndef NDEBUG
        const size_t repacked_size = ctx.kernels->rhs_info.packed_size(n, k, nr, kr, k_q4_0_block_size);
        GGML_ASSERT(repacked_size <= data_size && "repacked size larger than the packed size!");
#endif
        struct kai_rhs_pack_qs4cxs1s0_param params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        ctx.kernels->rhs_info.pack_func(1, n, k, nr, kr, sr, k_q4_0_block_size, (const uint8_t *)data, NULL, tensor->data, 0, &params);

        return 0;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::kleidiai

static void ggml_backend_cpu_kleidiai_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::kleidiai::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_kleidiai_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::kleidiai::tensor_traits *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_kleidiai_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_KLEIDIAI";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_kleidiai_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_kleidiai_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_kleidiai_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::kleidiai {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->type == GGML_TYPE_Q4_0 &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type(-1) && ctx.kernels
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type(-1)) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::kleidiai

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(int n_threads) {
    static ggml::cpu::kleidiai::extra_buffer_type ctx;
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_kleidiai = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_kleidiai_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_kleidiai_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_kleidiai_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &ctx,
    };

    init_kleidiai_context(n_threads);

    return &ggml_backend_cpu_buffer_type_kleidiai;
}
