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
#include <mutex>
#include <cstring>
#include <immintrin.h>
#include <kompute/Kompute.hpp>

#ifndef __STDC_IEC_559__
#error Your C implementation is not IEC 559 compliant, which is required for proper Vulkan interop.
#endif

#define MULTILINE_QUOTE(...) #__VA_ARGS__

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
    std::mutex tensors_mutex;
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
    printf("%s: Context: %p Name: '%s'\n", __func__, ctx, name);

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
    printf("%s: Context: %p Name: '%s'\n", __func__, ctx, name);

    auto res = ctx->buffers.find(name);
    if (res == ctx->buffers.end()) return nullptr;
    return res->second;
}


void ggml_vk_h2d_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    if (t->backend != GGML_BACKEND_GPU) {
        return;
    }

    auto data = t->data;
    auto size = ggml_nbytes(t);

    ctx->tensors_mutex.lock();
    auto res = ctx->tensors.find(t);
    ctx->tensors_mutex.unlock();

    if (res != ctx->tensors.end()) {
        assert(res->second->size() != size);
        res->second->setRawData(data);
        mgr.sequence()->eval<kp::OpTensorSyncDevice>({res->second});
    } else {
        std::vector<byte> vec(size);
        memcpy(vec.data(), data, size);

        auto tensor = mgr.tensorT<byte>(vec);
        mgr.sequence()->eval<kp::OpTensorSyncDevice>({tensor});
        ctx->tensors_mutex.lock();
        ctx->tensors.emplace(t, std::move(tensor));
        ctx->tensors_mutex.unlock();
    }
}

void ggml_vk_d2h_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    if (t->backend != GGML_BACKEND_GPU) {
        return;
    }

    auto data = t->data;
    auto size = ggml_nbytes(t);

    ctx->tensors_mutex.lock();
    auto res = ctx->tensors.find(t);
    ctx->tensors_mutex.unlock();
    assert(res != ctx->tensors.end());

    auto tensor = res->second;
    mgr.sequence()->eval<kp::OpTensorSyncLocal>({tensor});
    memcpy(data, tensor->data<void>(), size);
}

static
const std::shared_ptr<kp::Tensor> & ggml_vk_get_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    assert(t->backend != GGML_BACKEND_GPU);

    ctx->tensors_mutex.lock();
    auto res = ctx->tensors.find(t);
    ctx->tensors_mutex.unlock();
    assert(res != ctx->tensors.end());

    return res->second;
}


static std::vector<uint32_t> compileSource(const std::string& source) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> L(mutex);
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
#define GELU_COEF_A 0.044715;
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876;
)";


static const std::string program_dequantize_row_q4_0 =
        MULTILINE_QUOTE(
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
    const static auto spirv = compileSource(program_source_head+program_dequantize_row_q4_0);

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
        MULTILINE_QUOTE(
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
    const static auto spirv = compileSource(program_source_head+program_dequantize_row_q4_1);

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
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inBOff;
    uint outOff;
    uint row;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float inA[]; };
layout(binding = 1) buffer tensorInB { float inB[]; };
layout(binding = 2) buffer tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = inA[pcs.inAOff+i] MATH_OP inB[pcs.inBOff+(i ROW_OP)];
}
);

template<char mathOP>
void ggml_vk_abmath(kp::Sequence& seq,
                    const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                    const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                    const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                    uint32_t size, uint32_t row = 0) {
    const static auto spirv = compileSource(program_source_head+
                                            "#define MATH_OP "+std::string(1, mathOP)+"\n"
                                            "#define ROW_OP "+(row?"% pcs.row":"")+'\n'+
                                            program_abmath);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff, row;
    } pushConsts {
        inAOff, inBOff, outOff, row
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, inB, out}, spirv, {size}, {}, {pushConsts}));
}

template <typename... Args>
void ggml_vk_add(Args&&... args) {
    return ggml_vk_abmath<'+'>(std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_mul(Args&&... args) {
    return ggml_vk_abmath<'*'>(std::forward<Args>(args)...);
}


static const std::string program_scale =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inOff;
    float scale;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float in_[]; };
layout(binding = 1) buffer tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = in_[pcs.inOff+i] * pcs.scale;
}
);

void ggml_vk_scale(kp::Sequence& seq,
                   const std::shared_ptr<kp::Tensor>& in, uint32_t inOff,
                   const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                   uint32_t size, float scale) {
    const static auto spirv = compileSource(program_source_head+program_scale);

    struct PushConstants {
        uint32_t inOff, outOff;
        float scale;
    } pushConsts {
        inOff, outOff, scale
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({in, out}, spirv, {size}, {}, {pushConsts}));
}

void ggml_vk_xxlu(const std::vector<uint32_t>& spirv, kp::Sequence& seq,
                  const std::shared_ptr<kp::Tensor>& in, uint32_t inOff,
                  const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                  uint32_t size) {
    struct PushConstants {
        uint32_t inOff, outOff;
    } pushConsts {
        inOff, outOff
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({in, out}, spirv, {size}, {}, {pushConsts}));
}


static const std::string program_silu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float in_[]; };
layout(binding = 1) buffer tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const float x = in_[pcs.inOff+i];

    out_[pcs.outOff+i] = x / (1.0f + exp(-x));
}
);

template <typename... Args>
void ggml_vk_silu(Args&&... args) {
    const static auto spirv = compileSource(program_source_head+program_silu);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


static const std::string program_relu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float in_[]; };
layout(binding = 1) buffer tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = max(0.0, in_[pcs.inOff+i]);
}
);

template <typename... Args>
void ggml_vk_relu(Args&&... args) {
    const static auto spirv = compileSource(program_source_head+program_relu);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


static const std::string program_gelu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inAOff;
    uint inOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer tensorInA { float in_[]; };
layout(binding = 1) buffer tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const float x = in_[pcs.inOff+i];

    out_[pcs.outOff+i] = 0.5*x*(1.0 + tanh(SQRT_2_OVER_PI*x*(1.0 + GELU_COEF_A*x*x)));
}
);

template <typename... Args>
void ggml_vk_gelu(Args&&... args) {
    const static auto spirv = compileSource(program_source_head+program_gelu);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf) {
    printf("%s: evaluating graph\n", __func__);

    const int n_seq = gf->n_threads;

    std::vector<std::shared_ptr<kp::Sequence>> sequences(n_seq);

    for (auto& sequence : sequences) {
        sequence = mgr.sequence();
    }

    std::vector<std::thread> threads(n_seq);

    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int n_nodes_per_seq = (gf->n_nodes + n_seq - 1) / n_seq;

        threads[seq_idx] = std::thread([&, seq_idx, n_nodes_per_seq] () {
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            auto& seq = *sequences[seq_idx];

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

                switch (dst->op) {
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_TRANSPOSE:
                    case GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case GGML_OP_ADD:
                        {
                            ggml_vk_add(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ggml_nelements(dst));
                        } break;
                    case GGML_OP_MUL:
                        {
                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                ggml_vk_mul(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ne00, ggml_nelements(dst));
                            } else {
                                ggml_vk_mul(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ggml_nelements(dst));
                            }
                        } break;
                    case GGML_OP_SCALE:
                        {
                            const float scale = *(const float *) src1->data;
                            ggml_vk_scale(seq, id_src0, offs_src0, id_dst, offs_dst, ggml_nelements(dst), scale);
                        } break;
                    case GGML_OP_SILU:
                        {
                            ggml_vk_silu(seq, id_src0, offs_src0, id_dst, offs_dst, ggml_nelements(dst));
                        } break;
                    case GGML_OP_RELU:
                        {
                            ggml_vk_relu(seq, id_src0, offs_src0, id_dst, offs_dst, ggml_nelements(dst));
                        } break;
                    case GGML_OP_GELU:
                        {
                            ggml_vk_gelu(seq, id_src0, offs_src0, id_dst, offs_dst, ggml_nelements(dst));
                        } break;
                }
            }
        });
    }

    // Wait for all threads to finish
    for (auto& thread : threads) {
        if (thread.joinable())
            thread.join();
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
