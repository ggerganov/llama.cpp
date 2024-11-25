// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-tinyblas.h"
#include "ggml-backend-impl.h"

#include "sgemm.h"

#include <memory>
#include <cstring>

// TODO: see how to use threads/pool for all backend: ggml_graph_compute / ggml_threadpool
// https://github.com/ggerganov/llama.cpp/pull/1999
#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

namespace ggml::backend::tinyblas {

    static const char* NAME = "tinyBLAS";

    struct context {
        int n_threads = GGML_DEFAULT_N_THREADS;
        std::unique_ptr<char[]> work_data;
        size_t work_size = 0;
    };

    template<bool RUN>
    static bool mul_mat(int64_t m, int64_t n, int64_t k,
            const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
            int ith, int nth,
            const enum ggml_type Atype, const enum ggml_type Btype, const enum ggml_type Ctype)
    {
        GGML_ASSERT(Ctype == GGML_TYPE_F32);
        switch (Atype) {
        case GGML_TYPE_F32:
            if (Btype != GGML_TYPE_F32) return false;
            return gemm<RUN>(m, n, k, (const float*)A, lda, (const float*)B, ldb, (float*)C, ldc, ith, nth);
            break;
        case GGML_TYPE_F16:
            switch (Btype) {
            case GGML_TYPE_F32:
                return gemm<RUN>(m, n, k, (const ggml_fp16_t*)A, lda, (const float*)B, ldb, (float*)C, ldc, ith, nth);
            case GGML_TYPE_F16:
                return gemm<RUN>(m, n, k, (const ggml_fp16_t*)A, lda, (const ggml_fp16_t*)B, ldb, (float*)C, ldc, ith, nth);
            default:
                return false;
            }
            break;
        case GGML_TYPE_BF16:
            switch (Btype) {
            case GGML_TYPE_F32:
                return gemm<RUN>(m, n, k, (const ggml_bf16_t*)A, lda, (const float*)B, ldb, (float*)C, ldc, ith, nth);
            case GGML_TYPE_BF16:
                return gemm<RUN>(m, n, k, (const ggml_bf16_t*)A, lda, (const ggml_bf16_t*)B, ldb, (float*)C, ldc, ith, nth);
            default:
                return false;
            }
            break;
        case GGML_TYPE_Q8_0:
            if (Btype != GGML_TYPE_Q8_0) return false;
            return gemm<RUN>(m, n, k, (const block_q8_0*)A, lda, (const block_q8_0*)B, ldb, (float*)C, ldc, ith, nth);
            break;
        case GGML_TYPE_Q4_0:
            if (Btype != GGML_TYPE_Q8_0) return false;
            return gemm<RUN>(m, n, k, (const block_q4_0*)A, lda, (const block_q8_0*)B, ldb, (float*)C, ldc, ith, nth);
            break;
        case GGML_TYPE_Q5_0:
            if (Btype != GGML_TYPE_Q8_0) return false;
            return gemm<RUN>(m, n, k, (const block_q5_0*)A, lda, (const block_q8_0*)B, ldb, (float*)C, ldc, ith, nth);
            break;
        case GGML_TYPE_IQ4_NL:
            if (Btype != GGML_TYPE_Q8_0) return false;
            return gemm<RUN>(m, n, k, (const block_iq4_nl*)A, lda, (const block_q8_0*)B, ldb, (float*)C, ldc, ith, nth);
            break;
        default:
            return false;
        }
        return false;
    }

    static bool supports_mul_mat(ggml_backend_dev_t, const struct ggml_tensor * dst) {
        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        if (dst->type != GGML_TYPE_F32) return false;

        if (ne0 != ne01) return false;
        if (ne1 != ne11) return false;
        if (ne2 != ne12) return false;
        if (ne3 != ne13) return false;

        // we don't support permuted src0 or src1
        if (nb00 != ggml_type_size(src0->type)) return false;
        if (nb10 != ggml_type_size(src1->type)) return false;

        // dst cannot be transposed or permuted
        if (nb0 != sizeof(float)) return false;
        if (nb0 > nb1) return false;
        if (nb1 > nb2) return false;
        if (nb2 > nb3) return false;

        if (ggml_is_contiguous(src1)) {
            if (mul_mat<false>(ne01, ne11, ne00/ggml_blck_size(src0->type),
                    src0->data, nb01/ggml_type_size(src0->type),
                    src1->data, nb11/ggml_type_size(src1->type),
                    dst->data, nb1/ggml_type_size(dst->type),
                    0, 1, src0->type, src1->type, GGML_TYPE_F32)) {
                return true;
            }
        }

        // after convert B: FP32 => src0->vec_dot_type
        enum ggml_type const vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
        if ((src1->type != vec_dot_type) && (src1->type == GGML_TYPE_F32)) {
            if (mul_mat<false>(ne01, ne11, ne00/ggml_blck_size(src0->type),
                    src0->data, nb01/ggml_type_size(src0->type),
                    src1->data, nb11/ggml_type_size(src1->type),
                    dst->data, nb1/ggml_type_size(dst->type),
                    0, 1, src0->type, vec_dot_type, GGML_TYPE_F32)) {
                // TODO: how to resize work_data here
                return true;
            }
        }
        return false;
    }

    static void mul_mat(ggml::backend::tinyblas::context * ctx, struct ggml_tensor * dst, const int ith, const int nth) {
        const struct ggml_tensor * src0 = dst->src[0];
        const struct ggml_tensor * src1 = dst->src[1];

        GGML_TENSOR_BINARY_OP_LOCALS

        const enum ggml_type type0 = src0->type;
        const enum ggml_type type1 = src1->type;

        // broadcast factors
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        if (ggml_is_contiguous(src1)) {
            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    const void* data0 = (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03;
                    const void* data1 = (const char *)src1->data + i12*nb12 + i13*nb13;
                    void* data  = (char *)dst->data + i12*nb2 + i13*nb3;
                    if (!mul_mat<true>(ne01, ne11, ne00/ggml_blck_size(src0->type),
                            data0, nb01/ggml_type_size(src0->type),
                            data1, nb11/ggml_type_size(src1->type),
                            data, nb1/ggml_type_size(dst->type),
                            ith, nth, type0, type1, GGML_TYPE_F32)) {
                        goto UseGgmlGemm1;
                    }
                }
            }
            return;
        }
        UseGgmlGemm1:;

        // with B converted from FP32 -> vec_dot_type
        GGML_ASSERT(src1->type == GGML_TYPE_F32); // for use 'from_float'
        enum ggml_type    const vec_dot_type = ggml_get_type_traits_cpu(type0)->vec_dot_type;
        ggml_from_float_t const from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

        if (src1->type != vec_dot_type) {
            const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
            // const size_t row_size = ggml_row_size(vec_dot_type, ne10);
            const size_t nbw2 = nbw1*ne11;
            const size_t nbw3 = nbw2*ne12;

            // TODO: move to: supports_mul_mat
            if ((ith == 0) && (ctx->work_size < ne13*nbw3)) {
                ctx->work_data.reset(new char[ne13*nbw3]);
                ctx->work_size = ne13*nbw3;
            }
#ifdef GGML_USE_OPENMP
#pragma omp barrier
#else
            static_assert(false, "Not implemented: use GGML_USE_OPENMP");
#endif
            char * wdata = ctx->work_data.get();

            for (int64_t i13 = 0; i13 < ne13; ++i13) {
                for (int64_t i12 = 0; i12 < ne12; ++i12) {
                    for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                        from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                                   (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                   ne10);
                    }
                }
            }

            // synchronize all threads!
#ifdef GGML_USE_OPENMP
#pragma omp barrier
#else
            static_assert(false, "Not implemented: use GGML_USE_OPENMP");
#endif
            // mat-mul bis...
            for (int64_t i13 = 0; i13 < ne13; i13++)
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    const void* data0 = (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03;
                    const void* data1 = (const char *)wdata      + i12*nbw2    + i13*nbw3;

                    void* data  = (char *)dst->data + i12*nb2 + i13*nb3;
                    if (!mul_mat<true>(ne01, ne11, ne00/ggml_blck_size(src0->type),
                            data0, nb01/ggml_type_size(src0->type),
                            data1, nbw1/ggml_type_size(vec_dot_type),
                            data, nb1/ggml_type_size(dst->type),
                            ith, nth, type0, vec_dot_type, GGML_TYPE_F32)) {
                        goto UseGgmlGemm2;
                    }
                }
            return;
        }
        UseGgmlGemm2:;
    }

    static const char * get_name(ggml_backend_t /*backend*/) {
        return NAME;
    }

    static void free(ggml_backend_t backend) {
        context * ctx = (context *)backend->context;
        delete ctx;
        delete backend;
    }

    static enum ggml_status graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
        context * ctx = (context *)backend->context;

        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
            case GGML_OP_MUL_MAT:
#ifdef GGML_USE_OPENMP
#pragma omp parallel num_threads(ctx->n_threads)
            {
                int ith = omp_get_thread_num();
                int nth = ctx->n_threads;
                mul_mat(ctx, node, ith, nth);
            }
#else
            static_assert(false, "Not implemented: use GGML_USE_OPENMP");
            mul_mat(ctx, node, 0, 1);
#endif
            break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            }
        }

        return GGML_STATUS_SUCCESS;
    }

    static struct ggml_backend_i interface = {
            /* .get_name                = */ get_name,
            /* .free                    = */ free,
            /* .set_tensor_async        = */ NULL,
            /* .get_tensor_async        = */ NULL,
            /* .cpy_tensor_async        = */ NULL,
            /* .synchronize             = */ NULL,
            /* .graph_plan_create       = */ NULL,
            /* .graph_plan_free         = */ NULL,
            /* .graph_plan_update       = */ NULL,
            /* .graph_plan_compute      = */ NULL,
            /* .graph_compute           = */ graph_compute,
            /* .event_record            = */ NULL,
            /* .event_wait              = */ NULL,
    };

    static ggml_guid_t guid(void) {
        static ggml_guid guid = { 0x23, 0xf5, 0x9f, 0xa2, 0xb1, 0x48, 0x39, 0x25, 0x83, 0xcd, 0x79, 0x16, 0xb7, 0x23, 0x94, 0xde };
        return &guid;
    }

    static ggml_backend_t init(void) {
        context * ctx = new context;

        ggml_backend_t backend = new ggml_backend {
            /* .guid      = */ guid(),
                    /* .interface = */ interface,
                    /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_tinyblas_reg(), 0),
                    /* .context   = */ ctx,
        };

        return backend;
    }

    static bool is_tinyblas(ggml_backend_t backend) {
        return backend != NULL && ggml_guid_matches(backend->guid, guid());
    }

    static void set_n_threads(ggml_backend_t backend, int n_threads) {
        GGML_ASSERT(is_tinyblas(backend));
        context * ctx = (context *)backend->context;
        ctx->n_threads = n_threads;
    }

}

// device interface
namespace ggml::backend::tinyblas::device {
    static const char * get_name(ggml_backend_dev_t) {
        return "BLAS";
    }

    static const char * get_description(ggml_backend_dev_t) {
        return "tinyBLAS";
    }

    static void get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
        // TODO
        *free = 0;
        *total = 0;
    }

    static enum ggml_backend_dev_type get_type(ggml_backend_dev_t) {
        return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }

    static void get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
        props->name        = get_name(dev);
        props->description = get_description(dev);
        props->type        = get_type(dev);
        get_memory(dev, &props->memory_free, &props->memory_total);
        props->caps = {
                /* .async                 = */ false,
                /* .host_buffer           = */ false,
                /* .buffer_from_host_ptr  = */ true,
                /* .events                = */ false,
        };
    }

    static ggml_backend_t init_backend(ggml_backend_dev_t, const char *) {
        return ggml::backend::tinyblas::init();
    }

    static ggml_backend_buffer_type_t get_buffer_type(ggml_backend_dev_t) {
        return ggml_backend_cpu_buffer_type();
    }

    static ggml_backend_buffer_t buffer_from_host_ptr(ggml_backend_dev_t, void * ptr, size_t size, size_t) {
        return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    }

    static bool supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
        switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            return supports_mul_mat(device, op);
        default:
            return false;
        }
    }

    static bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
        return ggml_backend_buft_is_host(buft);
    }

    static const struct ggml_backend_device_i interface = {
            /* .get_name             = */ get_name,
            /* .get_description      = */ get_description,
            /* .get_memory           = */ get_memory,
            /* .get_type             = */ get_type,
            /* .get_props            = */ get_props,
            /* .init_backend         = */ init_backend,
            /* .get_buffer_type      = */ get_buffer_type,
            /* .get_host_buffer_type = */ NULL,
            /* .buffer_from_host_ptr = */ buffer_from_host_ptr,
            /* .supports_op          = */ supports_op,
            /* .supports_buft        = */ supports_buft,
            /* .offload_op           = */ NULL,
            /* .event_new            = */ NULL,
            /* .event_free           = */ NULL,
            /* .event_synchronize    = */ NULL,
    };

}

// backend reg interface
namespace ggml::backend::tinyblas::reg {
    static const char * get_name(ggml_backend_reg_t) {
        return ggml::backend::tinyblas::NAME;
    }

    static size_t get_device_count(ggml_backend_reg_t) {
        return 1;
    }

    static ggml_backend_dev_t get_device(ggml_backend_reg_t reg, size_t index) {
        GGML_ASSERT(index == 0);

        static ggml_backend_device device = {
                /* .iface   = */ ggml::backend::tinyblas::device::interface,
                /* .reg     = */ reg,
                /* .context = */ nullptr,
        };

        return &device;
    }

    static void * get_proc_address(ggml_backend_reg_t, const char * name) {
        if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
            return (void *)ggml::backend::tinyblas::set_n_threads;
        }
        return NULL;
    }

    static const struct ggml_backend_reg_i interface = {
            /* .get_name         = */ get_name,
            /* .get_device_count = */ get_device_count,
            /* .get_device       = */ get_device,
            /* .get_proc_address = */ get_proc_address,
    };

}

ggml_backend_reg_t ggml_backend_tinyblas_reg(void) {
    static struct ggml_backend_reg backend_reg = {
            /* .iface   = */ ggml::backend::tinyblas::reg::interface,
            /* .context = */ NULL,
    };
    return &backend_reg;
}
