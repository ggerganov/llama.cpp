#include "ggml-vulkan.h"
#include "ggml.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <exception>
#include <thread>
#include <cstring>
#include <immintrin.h>
#include <kompute/Kompute.hpp>

#ifndef __STDC_IEC_559__
#error Your C implementation is not IEC 559 compliant, which is required for proper Vulkan interop.
#endif

#define MULTILINE_QUOTE(...) #__VA_ARGS__
#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x

#define QK4_0 32
#define QR4_0 2
#define QK4_1 32


typedef ggml_fp16_t half;
enum class byte : unsigned char {};

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half d;
    half m;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;

struct ggml_kompute_context {
    std::unordered_map<const char *, std::shared_ptr<kp::Tensor>> buffers;
    std::unordered_map<struct ggml_tensor *, std::shared_ptr<kp::Tensor>> tensors;
};


kp::Manager mgr;


ggml_kompute_context *ggml_vk_init() {
    return new ggml_kompute_context;
}

void ggml_metal_free(struct ggml_kompute_context * ctx) {
    delete ctx;
}


bool ggml_vk_add_buffer(
      struct ggml_kompute_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size,
                           size_t   max_size) {
    try {
        std::vector<byte> vec(std::max(size, max_size));
        std::memcpy(vec.data(), data, size);
        auto tensor = mgr.tensorT<byte>(vec);
        ctx->buffers.emplace(name, std::move(tensor));
    } catch (const std::exception & e) {
        fprintf(stderr, "ggml_vk: failed to add buffer '%s': %s\n", name, e.what());
        return false;
    }
    return true;
}

static
std::shared_ptr<kp::Tensor> ggml_vk_get_buffer(struct ggml_kompute_context * ctx, const char * name) {
    auto res = ctx->buffers.find(name);
    if (res == ctx->buffers.end()) return nullptr;
    return res->second;
}


void ggml_vk_h2d_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    if (t->backend != GGML_BACKEND_GPU) {
        return;
    }

    auto data = t->data;
    auto size = ggml_nbytes(t);

    std::vector<byte> vec(size);
    memcpy(vec.data(), data, size);

    auto tensor = mgr.tensorT<byte>(vec);
    mgr.sequence()->eval<kp::OpTensorSyncDevice>({tensor});
    ctx->tensors.emplace(t, std::move(tensor));
}

void ggml_vk_d2h_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    if (t->backend != GGML_BACKEND_GPU) {
        return;
    }

    auto data = t->data;
    auto size = ggml_nbytes(t);

    auto res = ctx->tensors.find(t);
    assert(res != ctx->tensors.end());

    auto tensor = res->second;
    mgr.sequence()->eval<kp::OpTensorSyncLocal>({tensor});
    memcpy(data, tensor->data<void>(), size);
}

static
const std::shared_ptr<kp::Tensor> & ggml_vk_get_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    assert(t->backend != GGML_BACKEND_GPU);

    auto res = ctx->tensors.find(t);
    assert(res != ctx->tensors.end());

    return res->second;
}


static std::vector<uint32_t> compileSource(const std::string& source) {
    //FIXME: Terrible solution!!!!
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}


template<class T>
std::vector<half> getVecBlockQ4_0D(T *x, unsigned nb) {
    std::vector<half> fres(nb);
    for (unsigned it = 0; it != nb; it++) {
        fres[it] = x[it].d;
    }
    return fres;
}

template<class T>
std::vector<half> getVecBlockQ4_0M(T *x, unsigned nb) {
    std::vector<half> fres(nb);
    for (unsigned it = 0; it != nb; it++) {
        fres[it] = x[it].m;
    }
    return fres;
}

template<class T>
std::vector<uint8_t> getVecBlockQ4_0QS(T *x, unsigned nb, unsigned qk) {
    std::vector<uint8_t> fres(nb*(qk/2));
    for (unsigned x_it = 0; x_it != nb; x_it++) {
        for (unsigned qs_it = 0; qs_it != qk / 2; qs_it++) {
            fres[x_it * (qk / 2) + qs_it] = x[x_it].qs[qs_it];
        }
    }
    return fres;
};


static const std::string program_source_head = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#define QK4_0 32
#define QR4_0 2
#define QK4_1 32
)";


static const std::string program_dequantize_row_q4_0 =
        program_source_head+'\n'+MULTILINE_QUOTE(
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) buffer tensorBlockQ4_0D { float16_t x_d[]; };
layout(binding = 1) buffer tensorBlockQ4_0QS { uint8_t x_qs[]; };
layout(binding = 2) buffer tensorY { float y[]; };

void main() {
    const int qk = QK4_0;

    const int i = int(gl_GlobalInvocationID.x);
    const int j = int(gl_GlobalInvocationID.y);

    const float d = float(x_d[i]);
    const uint8_t qs = x_qs[i * (qk / 2) + j];

    const int x0 = (qs & 0x0F) - 8;
    const int x1 = (qs >>   4) - 8;

    y[i*qk + j + 0   ] = float(x0)*d;
    y[i*qk + j + qk/2] = float(x1)*d;
}
);

void ggml_vk_dequantize_row_q4_0(const void *x_, float *y, int k) {
    static const int qk = QK4_0;
    const unsigned nb = k / qk;
    const unsigned y_size = nb*qk;
    const static auto spirv = compileSource(program_dequantize_row_q4_0);

    const auto x = reinterpret_cast<const block_q4_0*>(x_);

    assert(k % qk == 0);

    const auto tensorBlockQ4_0D = mgr.tensorT<half>(getVecBlockQ4_0D(x, nb));
    const auto tensorBlockQ4_0QS = mgr.tensorT<uint8_t>(getVecBlockQ4_0QS(x, nb, qk));
    const auto tensorY = mgr.tensor(std::vector<float>(y, y+y_size));

    mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({tensorBlockQ4_0D, tensorBlockQ4_0QS, tensorY})
            ->record<kp::OpAlgoDispatch>(mgr.algorithm({tensorBlockQ4_0D, tensorBlockQ4_0QS, tensorY}, spirv, {nb, qk/2, 0}))
            ->record<kp::OpTensorSyncLocal>({tensorY})
            ->eval();

    std::memcpy(y, tensorY->data(), tensorY->size()*sizeof(*y));
}


static const std::string program_dequantize_row_q4_1 =
        program_source_head+'\n'+MULTILINE_QUOTE(
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) buffer tensorBlockQ4_0D { float16_t x_d[]; };
layout(binding = 1) buffer tensorBlockQ4_0M { float16_t x_m[]; };
layout(binding = 2) buffer tensorBlockQ4_0QS { uint8_t x_qs[]; };
layout(binding = 3) buffer tensorY { float y[]; };

void main() {
    const int qk = QK4_1;

    const int i = int(gl_GlobalInvocationID.x);
    const int j = int(gl_GlobalInvocationID.y);

    const float d = float(x_d[i]);
    const float m = float(x_m[i]);
    const uint8_t qs = x_qs[i * (qk / 2) + j];

    const int x0 = (qs & 0x0F);
    const int x1 = (qs >>   4);

    y[i*qk + j + 0   ] = x0*d + m;
    y[i*qk + j + qk/2] = x1*d + m;
}
);

void ggml_vk_dequantize_row_q4_1(const void *x_, float *y, int k) {
    static const int qk = QK4_1;
    const unsigned nb = k / qk;
    const unsigned y_size = nb*qk;
    const static auto spirv = compileSource(program_dequantize_row_q4_1);

    const auto x = reinterpret_cast<const block_q4_1*>(x_);

    assert(k % qk == 0);

    const auto tensorBlockQ4_0D = mgr.tensorT<half>(getVecBlockQ4_0D(x, nb));
    const auto tensorBlockQ4_0M = mgr.tensorT<half>(getVecBlockQ4_0M(x, nb));
    const auto tensorBlockQ4_0QS = mgr.tensorT<uint8_t>(getVecBlockQ4_0QS(x, nb, qk));
    const auto tensorY = mgr.tensor(std::vector<float>(y, y+y_size));

    mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({tensorBlockQ4_0D, tensorBlockQ4_0M, tensorBlockQ4_0QS, tensorY})
            ->record<kp::OpAlgoDispatch>(mgr.algorithm({tensorBlockQ4_0D, tensorBlockQ4_0M, tensorBlockQ4_0QS, tensorY}, spirv, {nb, qk/2, 0}))
            ->record<kp::OpTensorSyncLocal>({tensorY})
            ->eval();

    std::memcpy(y, tensorY->data(), tensorY->size()*sizeof(*y));
}


static const std::string program_abmath =
        program_source_head+'\n'+MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inBOff;
    uint outOff;
} pcs;


layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float inA[]; };
layout(binding = 1) buffer tensorInB { float inB[]; };
layout(binding = 2) buffer tensorout { float out[]; };


void main() {
    const int i = int(gl_GlobalInvocationID.x);

    out[pcs.outOff+i] = inA[pcs.inAOff+i] MATH_OP inB[pcs.inBOff+i];
}
);

template<char mathOP>
void ggml_vk_abmath(const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                 const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                 std::shared_ptr<kp::Tensor>& out, uint32_t outOff) {
    const static auto spirv = compileSource("#define MATH_OP "+std::string(1, mathOP)+'\n'+program_abmath);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
    } pushConsts {
        inAOff, inBOff, outOff
    };

    mgr.sequence()
            ->eval<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, inB, out}, spirv, {std::min(inA->size()-inAOff, inB->size()-inBOff)}, {}, {pushConsts}));
}

template <typename... Args>
void ggml_vk_add(Args&&... args) {
    return ggml_vk_abmath<'+'>(std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_mul(Args&&... args) {
    return ggml_vk_abmath<'*'>(std::forward<Args>(args)...);
}


void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf) {
    printf("%s: evaluating graph\n", __func__);

    const int n_seq = gf->n_threads;

    std::vector<kp::Sequence> sequences(n_seq);

    std::vector<std::thread> threads(n_seq);

    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int n_nodes_per_seq = (gf->n_nodes + n_seq - 1) / n_seq;

        threads[seq_idx] = std::thread([&, seq_idx, n_nodes_per_seq] () {
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            auto& seq = sequences[seq_idx];

            const int node_start = (seq_idx + 0) * n_nodes_per_seq;
            const int node_end = (seq_idx == n_seq - 1) ? gf->n_nodes : (seq_idx + 1) * n_nodes_per_seq;

            for (int i = node_start; i < node_end; ++i) {
                printf("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

                struct ggml_tensor * src0 = gf->nodes[i]->src0;
                struct ggml_tensor * src1 = gf->nodes[i]->src1;
                struct ggml_tensor * dst = gf->nodes[i];

                const int64_t ne00 = src0 ? src0->ne[0] : 0;
                const int64_t ne01 = src0 ? src0->ne[1] : 0;
                const int64_t ne02 = src0 ? src0->ne[2] : 0;
                const int64_t ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t ne10 = src1 ? src1->ne[0] : 0;
                const int64_t ne11 = src1 ? src1->ne[1] : 0;
                const int64_t ne12 = src1 ? src1->ne[2] : 0;
                const int64_t ne13 = src1 ? src1->ne[3] : 0;  (void)ne13;

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; (void)nb13;

                const int64_t ne0 = dst ? dst->ne[0] : 0;
                const int64_t ne1 = dst ? dst->ne[1] : 0;
                const int64_t ne2 = dst ? dst->ne[2] : 0;
                const int64_t ne3 = dst ? dst->ne[3] : 0;

                const uint64_t nb0 = dst ? dst->nb[0] : 0;
                const uint64_t nb1 = dst ? dst->nb[1] : 0;
                const uint64_t nb2 = dst ? dst->nb[2] : 0;
                const uint64_t nb3 = dst ? dst->nb[3] : 0;

                const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
                const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
                const enum ggml_type dstt = dst ? dst->type : GGML_TYPE_COUNT;

                std::shared_ptr<kp::Tensor> id_src0 = src0 ? ggml_vk_get_tensor(ctx, src0) : nullptr;
                std::shared_ptr<kp::Tensor> id_src1 = src1 ? ggml_vk_get_tensor(ctx, src1) : nullptr;
                std::shared_ptr<kp::Tensor> id_dst  = dst  ? ggml_vk_get_tensor(ctx, dst)  : nullptr;
            }
        });
    }
}


template<>
kp::Tensor::TensorDataTypes
kp::TensorT<half>::dataType()
{
    return TensorDataTypes::eFloat;
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<uint8_t>::dataType()
{
    return TensorDataTypes::eUnsignedInt;
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<byte>::dataType()
{
    return TensorDataTypes::eUnsignedInt;
}
