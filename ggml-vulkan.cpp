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
#include <atomic>
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

void ggml_vk_free(struct ggml_kompute_context * ctx) {
    delete ctx;
}


size_t ggml_vk_mem_used(struct ggml_kompute_context * ctx) {
    size_t fres = 0;
    ctx->tensors_mutex.lock();
    for (const auto& tensor : ctx->tensors) {
        fres += tensor.second->size();
    }
    ctx->tensors_mutex.unlock();
    for (const auto& buffer : ctx->buffers) {
        fres += buffer.second->size();
    }
    return fres;
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
kp::Tensor* ggml_vk_get_buffer(struct ggml_kompute_context * ctx, const char * name) {
    printf("%s: Context: %p Name: '%s'\n", __func__, ctx, name);

    const auto res = ctx->buffers.find(name);
    if (res == ctx->buffers.end()) return nullptr;
    return res->second.get();
}


void ggml_vk_h2d_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    const auto data = t->data;
    const auto size = ggml_nbytes(t);

    ctx->tensors_mutex.lock();
    const auto res = ctx->tensors.find(t);

    if (res != ctx->tensors.end()) {
        ctx->tensors_mutex.unlock();
        GGML_ASSERT(res->second->size() != size);
        res->second->setRawData(data);
        mgr.sequence()->eval<kp::OpTensorSyncDevice>({res->second});
        printf("%s: Updating Host->GPU tensor: %p\n", __func__, t);
    } else {
        std::vector<byte> vec(size);
        memcpy(vec.data(), data, size);

        auto tensor = mgr.tensorT<byte>(vec);
        mgr.sequence()->eval<kp::OpTensorSyncDevice>({tensor});
        ctx->tensors.emplace(t, std::move(tensor));
        ctx->tensors_mutex.unlock();
        printf("%s: Creating Host->GPU tensor: %p\n", __func__, t);
    }
}

void ggml_vk_d2h_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    const auto data = t->data;
    const auto size = ggml_nbytes(t);

    ctx->tensors_mutex.lock();
    const auto res = ctx->tensors.find(t);
    ctx->tensors_mutex.unlock();
    GGML_ASSERT(res != ctx->tensors.end());

    auto& tensor = res->second;
    mgr.sequence()->eval<kp::OpTensorSyncLocal>({tensor});
    memcpy(data, tensor->data<void>(), size);
    printf("%s: Updating GPU->Host tensor: %p\n", __func__, t);
}

static
const std::shared_ptr<kp::Tensor> & ggml_vk_get_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    printf("%s: Context: %p Tensor: %p\n", __func__, ctx, t);

    ctx->tensors_mutex.lock();
    const auto res = ctx->tensors.find(t);
    const auto end = ctx->tensors.end();
    ctx->tensors_mutex.unlock();

    if (res == end) {
        ggml_vk_h2d_tensor(ctx, t);
        return ggml_vk_get_tensor(ctx, t);
    }

    return res->second;
}


static std::vector<uint32_t> glsl_compile_source(const std::string& source, const char *debug_name) {
    printf("%s: Compiling compute program: %s\n", __func__, debug_name);
    static std::mutex mutex;
    std::lock_guard<std::mutex> L(mutex);
    //FIXME: Terrible solution!!!!
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv > /dev/null").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}


template<class T>
std::vector<half> get_vec_block_Q4_0D(T *x, unsigned nb) {
    std::vector<half> fres(nb);
    for (unsigned it = 0; it != nb; it++) {
        fres[it] = x[it].d;
    }
    return fres;
}

template<class T>
std::vector<half> get_vec_block_Q4_0M(T *x, unsigned nb) {
    std::vector<half> fres(nb);
    for (unsigned it = 0; it != nb; it++) {
        fres[it] = x[it].m;
    }
    return fres;
}

template<class T>
std::vector<uint8_t> get_vec_block_Q4_0QS(T *x, unsigned nb, unsigned qk) {
    std::vector<uint8_t> fres(nb*(qk/2));
    for (unsigned x_it = 0; x_it != nb; x_it++) {
        for (unsigned qs_it = 0; qs_it != qk / 2; qs_it++) {
            fres[x_it * (qk / 2) + qs_it] = x[x_it].qs[qs_it];
        }
    }
    return fres;
};


static const std::string program_source_head = R"(#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: enable
#extension GL_EXT_control_flow_attributes: enable

#define QK4_0 32
#define QR4_0 2
#define QK4_1 32

#define GELU_COEF_A 0.044715
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876

#ifndef QK_K
#define QK_K 256
#endif

#if QK_K == 256
#define K_SCALE_SIZE 12
#else
#define K_SCALE_SIZE 4
#endif

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
)";


static const std::string program_dequantize_row_q4_0 =
        MULTILINE_QUOTE(
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0) buffer restrict readonly tensorBlockQ4_0D { float16_t x_d[]; };
layout(binding = 1) buffer restrict readonly tensorBlockQ4_0QS { uint8_t x_qs[]; };
layout(binding = 2) buffer restrict writeonly tensorY { float y[]; };

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
    const static auto spirv = glsl_compile_source(program_source_head+program_dequantize_row_q4_0, __func__);

    const auto x = reinterpret_cast<const block_q4_0*>(x_);

    GGML_ASSERT(k % qk == 0);

    const auto tensorBlockQ4_0D = mgr.tensorT<half>(get_vec_block_Q4_0D(x, nb));
    const auto tensorBlockQ4_0QS = mgr.tensorT<uint8_t>(get_vec_block_Q4_0QS(x, nb, qk));
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
layout(binding = 0) buffer restrict readonly tensorBlockQ4_0D { float16_t x_d[]; };
layout(binding = 1) buffer restrict readonly tensorBlockQ4_0M { float16_t x_m[]; };
layout(binding = 2) buffer restrict readonly tensorBlockQ4_0QS { uint8_t x_qs[]; };
layout(binding = 3) buffer restrict writeonly tensorY { float y[]; };

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
    const static auto spirv = glsl_compile_source(program_source_head+program_dequantize_row_q4_1, __func__);

    const auto x = reinterpret_cast<const block_q4_1*>(x_);

    GGML_ASSERT(k % qk == 0);

    const auto tensorBlockQ4_0D = mgr.tensorT<half>(get_vec_block_Q4_0D(x, nb));
    const auto tensorBlockQ4_0M = mgr.tensorT<half>(get_vec_block_Q4_0M(x, nb));
    const auto tensorBlockQ4_0QS = mgr.tensorT<uint8_t>(get_vec_block_Q4_0QS(x, nb, qk));
    const auto tensorY = mgr.tensor(std::vector<float>(y, y+y_size));

    mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({tensorBlockQ4_0D, tensorBlockQ4_0M, tensorBlockQ4_0QS, tensorY})
            ->record<kp::OpAlgoDispatch>(mgr.algorithm({tensorBlockQ4_0D, tensorBlockQ4_0M, tensorBlockQ4_0QS, tensorY}, spirv, {nb, qk/2, 0}))
            ->record<kp::OpTensorSyncLocal>({tensorY})
            ->eval();

    std::memcpy(y, tensorY->data(), tensorY->size()*sizeof(*y));
}


static const std::string program_fpx_to_fpx =
    MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inOff;
    uint outOff;
    uint row;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorIn { IN_TYPE in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { OUT_TYPE out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff + i] = OUT_TYPE(in_[pcs.inOff + i]);
}
);

void ggml_vk_fp32_to_fp16_row(kp::Sequence& seq,
                              const std::shared_ptr<kp::Tensor>& in, uint32_t inOff,
                              const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                              uint32_t size) {
    const static auto spirv = glsl_compile_source(program_source_head+
                                                      "#define IN_TYPE float\n"
                                                      "#define OUT_TYPE float16_t\n"+
                                                      program_fpx_to_fpx, __func__);

    struct PushConstants {
        uint32_t inOff, outOff;
    } const pushConsts {
        inOff, outOff
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({in, out}, spirv, {size}, {}, {pushConsts}));
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
layout(binding = 0) buffer restrict readonly tensorInA { float inA[]; };
layout(binding = 1) buffer restrict readonly tensorInB { float inB[]; };
layout(binding = 2) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = inA[pcs.inAOff+i] MATH_OP inB[pcs.inBOff+(i ROW_OP)];
}
);

template<char mathOP, bool with_row = false>
void ggml_vk_abmath(kp::Sequence& seq,
                    const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                    const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                    const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                    uint32_t size, uint32_t row = 0) {
    GGML_ASSERT(with_row?row:!row);

    const static auto spirv = glsl_compile_source(program_source_head+
                                            "#define MATH_OP "+std::string(1, mathOP)+"\n"
                                            "#define ROW_OP "+(with_row?"% pcs.row":"")+'\n'+
                                            program_abmath, __func__);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff, row;
    } const pushConsts {
        inAOff, inBOff, outOff, row
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, inB, out}, spirv, {size}, {}, {pushConsts}));
}


static const std::string program_scale =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inOff;
    uint outOff;
    float scale;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorIn { float in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = in_[pcs.inOff+i] * pcs.scale;
}
);

void ggml_vk_scale(kp::Sequence& seq,
                   const std::shared_ptr<kp::Tensor>& in, uint32_t inOff,
                   const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                   uint32_t size, float scale) {
    const static auto spirv = glsl_compile_source(program_source_head+program_scale, __func__);

    struct PushConstants {
        uint32_t inOff, outOff;
        float scale;
    } const pushConsts {
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
    } const pushConsts {
        inOff, outOff
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({in, out}, spirv, {size}, {}, {pushConsts}));
}


static const std::string program_silu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inOff;
    uint outOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorInA { float in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const float x = in_[pcs.inOff+i];

    out_[pcs.outOff+i] = x / (1.0f + exp(-x));
}
);

template <typename... Args>
void ggml_vk_silu(Args&&... args) {
    const static auto spirv = glsl_compile_source(program_source_head+program_silu, __func__);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


static const std::string program_relu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inOff;
    uint outOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorInA { float in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;

    out_[pcs.outOff+i] = max(0.0, in_[pcs.inOff+i]);
}
);

template <typename... Args>
void ggml_vk_relu(Args&&... args) {
    const static auto spirv = glsl_compile_source(program_source_head+program_relu, __func__);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


static const std::string program_gelu =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint inOff;
    uint outOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorInA { float in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint i = gl_GlobalInvocationID.x;
    const float x = in_[pcs.inOff+i];

    out_[pcs.outOff+i] = 0.5*x*(1.0 + tanh(SQRT_2_OVER_PI*x*(1.0 + GELU_COEF_A*x*x)));
}
);

template <typename... Args>
void ggml_vk_gelu(Args&&... args) {
    const static auto spirv = glsl_compile_source(program_source_head+program_gelu, __func__);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}


static const std::string program_soft_max =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint64_t ne00;
    uint64_t ne01;
    uint64_t ne02;
    uint inOff;
    uint outOff;
} pcs;

layout(local_size_x = nth) in;
layout(binding = 0) buffer restrict readonly tensorInA { float in_[]; };
layout(binding = 1) buffer restrict writeonly tensorOut { float out_[]; };

shared float buf[nth];

void main() {
    const uint64_t i03 = uint64_t(gl_GlobalInvocationID.z);
    const uint64_t i02 = uint64_t(gl_GlobalInvocationID.y);
    const uint64_t i01 = uint64_t(gl_GlobalInvocationID.x);

    const uint extra_off = uint(i03*pcs.ne02*pcs.ne01*pcs.ne00 + i02*pcs.ne01*pcs.ne00 + i01*pcs.ne00);
    const uint in_off = pcs.inOff + extra_off;
    const uint out_off = pcs.outOff + extra_off;

    // parallel max
    buf[gl_WorkGroupID.x] = uintBitsToFloat(0xFF800000);
    for (uint i00 = gl_WorkGroupID.x; i00 < pcs.ne00; i00 += nth) {
        buf[gl_WorkGroupID.x] = max(buf[gl_WorkGroupID.x], in_[in_off + i00]);
    }

    // reduce
    barrier();
    memoryBarrierShared();
    [[unroll]] for (uint i = nth/2; i > 0; i /= 2) {
        if (gl_WorkGroupID.x < i) {
            buf[gl_WorkGroupID.x] = max(buf[gl_WorkGroupID.x], buf[gl_WorkGroupID.x + i]);
        }
        barrier();
        memoryBarrierShared();
    }

    // broadcast (no effect?)
    if (gl_WorkGroupID.x == 0) {
        buf[0] = buf[0]; // ???
    }

    barrier();
    memoryBarrierShared();

    const float max_ = buf[0];

    // parallel sum
    buf[gl_WorkGroupID.x] = 0.0;
    for (uint i00 = gl_WorkGroupID.x; i00 < pcs.ne00; i00 += nth) {
        buf[gl_WorkGroupID.x] += exp(in_[in_off + i00] - max_);
    }

    // reduce
    barrier();
    memoryBarrierShared();
    for (uint i = nth/2; i > 0; i /= 2) {
        if (gl_WorkGroupID.x < i) {
            buf[gl_WorkGroupID.x] += buf[gl_WorkGroupID.x + i];
        }
        barrier();
        memoryBarrierShared();
    }

    // broadcast (no effect?)
    if (gl_WorkGroupID.x == 0) {
        buf[0] = buf[0]; // ???
    }

    barrier();
    memoryBarrierShared();

    const float sum = buf[0];

    for (uint i00 = gl_WorkGroupID.x; i00 < pcs.ne00; i00 += nth) {
        out_[out_off + i00] = exp(in_[in_off + i00] - max_) / sum;
    }
}
);

void ggml_vk_soft_max(kp::Sequence& seq,
                      const std::shared_ptr<kp::Tensor>& in, uint32_t inOff,
                      const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                      int64_t ne00, int64_t ne01, int64_t ne02, uint64_t ne03) {
    const static unsigned nth = 32;
    const static auto spirv = glsl_compile_source(program_source_head+"#define nth "+std::to_string(nth)+"\n"+program_soft_max, __func__);

    struct PushConstants {
        int64_t ne00, ne01, ne02;
        uint32_t inOff, outOff;
    } pushConsts {
        ne00, ne01, ne02, inOff, outOff
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({in, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts}));
}


static const std::string program_diag_mask_inf =
        MULTILINE_QUOTE(
layout(push_constant) uniform PushConstants {
    uint64_t ne00;
    uint64_t ne01;
    uint inAOff;
    uint inBOff;
    uint outOff;
} pcs;

layout(local_size_x = 1) in;
layout(binding = 0) buffer restrict readonly tensorInA { float inA[]; };
layout(binding = 1) buffer restrict readonly tensorInB { int inB[]; };
layout(binding = 2) buffer restrict writeonly tensorOut { float out_[]; };

void main() {
    const uint64_t i02 = uint64_t(gl_GlobalInvocationID.z);
    const uint64_t i01 = uint64_t(gl_GlobalInvocationID.y);
    const uint64_t i00 = uint64_t(gl_GlobalInvocationID.x);

    const int n_past = inB[pcs.inBOff];

    if (i00 > n_past + i01) {
        out_[uint(i02*pcs.ne01*pcs.ne00 + i01*pcs.ne00 + i00 + pcs.outOff)] = uintBitsToFloat(0xFF800000);
    } else {
        out_[uint(i02*pcs.ne01*pcs.ne00 + i01*pcs.ne00 + i00 + pcs.outOff)] = inA[uint(i02*pcs.ne01*pcs.ne00 + i01*pcs.ne00 + i00 + pcs.inAOff)];
    }
}
);

void ggml_vk_diag_mask_inf(kp::Sequence& seq,
                           const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                           const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                           const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                           int64_t ne00, int64_t ne01, int64_t ne02) {
    const static auto spirv = glsl_compile_source(program_source_head+program_diag_mask_inf, __func__);

    struct PushConstants {
        int64_t ne00, ne01;
        uint32_t inAOff, inBOff, outOff;
    } pushConsts {
        ne00, ne01, inAOff, inBOff, outOff
    };

    seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, inB, out}, spirv, {unsigned(ne00), unsigned(ne01), unsigned(ne02)}, {}, {pushConsts}));
}


static const std::string program_mul_mat_f16 =
    MULTILINE_QUOTE(
layout(local_size_x = (BM * BN) / (TM * TN), local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer tensorInA { float16_t inA[]; };
layout (binding = 1) readonly buffer tensorInB { float16_t inB[]; };
layout (binding = 2) writeonly buffer tensorOut { float out_[]; };

layout (push_constant) uniform parameter {
    int M;
    int N;
    int K;
    int inAStride;
    int inBStride;
    int outStride;
    uint inAOff;
    uint inBOff;
    uint outOff;
} pcs;

shared float16_t bufA[BM * (BK+1)];
shared float16_t bufB[BN * (BK+1)];

void main() {
    const int ir = int(gl_WorkGroupID.x);
    const int ic = int(gl_WorkGroupID.y);

    const int rstride = BM / TM;

    const int lr = int(gl_LocalInvocationID.x % rstride);
    const int lc = int(gl_LocalInvocationID.x / rstride);

    const int loadr = int(gl_LocalInvocationID.x % BK);
    const int loadc = int(gl_LocalInvocationID.x / BK);

    const int loadstride = int(gl_WorkGroupSize.x);

    int posA = ir * BM * pcs.inAStride;
    int posB = ic * BN * pcs.inBStride;

    float sums[TM * TN];
    float16_t cacheA[TM];
    float16_t cacheB[TN];

    [[unroll]] for (int i = 0; i < TM*TN; i++) {
        sums[i] = 0.0hf;
    }

    [[unroll]] for (int block = 0; block < pcs.K; block += BK) {
        [[unroll]] for (int l = 0; l < BM * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            bufA[(loadc + lc) * (BK+1) + loadr + lr] = inA[posA + (loadc + lc) * pcs.inAStride + loadr + lr];
        }
        [[unroll]] for (int l = 0; l < BN * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            bufB[(loadc + lc) * (BK+1) + loadr + lr] = inB[posB + (loadc + lc) * pcs.inBStride + loadr + lr];
        }

        barrier();

        posA += BK;
        posB += BK;

        [[unroll]] for (int i = 0; i < BK; i++) {
            // Load from shared into cache
            [[unroll]] for (int j = 0; j < BM; j++) {
                cacheA[j] = bufA[(lr + j*rstride) * (BK+1) + i];
            }
            [[unroll]] for (int j = 0; j < TN; j++) {
                cacheB[j] = bufB[(lc * TN + j) * (BK+1) + i];
            }

            [[unroll]] for (int cc = 0; cc < TN; cc++) {
                [[unroll]] for (int cr = 0; cr < TM; cr++) {
                    sums[cc * TM + cr] += float(cacheA[cr]) * float(cacheB[cc]);
                }
            }
        }

        barrier();
    }

    const int dr = ir * BM + lr;
    const int dc = ic * BN + lc * TN;

    [[unroll]] for (int cc = 0; cc < TN; cc++) {
        [[unroll]] for (int cr = 0; cr < TM; cr++) {
            out_[(dc + cc) * pcs.outStride + dr + cr*rstride] = sums[cc * TM + cr];
        }
    }
}
);

void ggml_vk_mul_mat_f16(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                         const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                         const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                         int64_t ne00, int64_t ne01, int64_t ne02, uint64_t ne03,
                         int64_t ne10, int64_t ne11,
                         int nb10, int nb11, int nb12, int nb13,
                         int nb2, int nb3) {
    const static auto spirv = glsl_compile_source(program_source_head+program_mul_mat_f16, __func__);

    const bool inB_cont_rows = nb10 == sizeof(float);
    const bool inB_cont_cols = (size_t)nb11 == ne11 * sizeof(float);

    struct PushConstants {
        int32_t M, N, K, inAStride, inBStride, outStride;
        uint32_t inAOff, inBOff, outOff;
    } pushConsts {
        (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01,
        inAOff, inBOff, outOff
    };

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            auto tmp = mgr.tensorT<half>(std::vector<half>(ne10*ne11));

            if (inB_cont_rows) {
                if (inB_cont_cols) {
                    ggml_vk_fp32_to_fp16_row(seq, inB, (i03*nb13 + i02*nb12)/sizeof(float), tmp, 0, ne10*ne11);
                }
                else {
                    for (int64_t i01 = 0; i01 < ne11; i01++) {
                        ggml_vk_fp32_to_fp16_row(seq, inB, (i03*nb13 + i02*nb12 + i01*nb11)/sizeof(float), tmp, i01*ne10, ne10);
                    }
                }
            } else {
                for (int64_t i01 = 0; i01 < ne11; i01++) {
                    for (int64_t i00 = 0; i00 < ne10; i00++) {
                        // Extremely slow because of single shader invocation
                        ggml_vk_fp32_to_fp16_row(seq, inB, (i03*nb13 + i02*nb12 + i01*nb11 + i00*nb10)/sizeof(float), tmp, i01*ne10 + i00, 1);
                    }
                }
            }

            seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, tmp, out}, spirv, {uint32_t(ne01/128), uint32_t(ne11/128)}, {}, {pushConsts}));
        }
    }
}


static const std::string program_mul_mat_f32 =
    MULTILINE_QUOTE(
layout(local_size_x = (BM * BN) / (TM * TN), local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer tensorInA { float inA[]; };
layout (binding = 1) readonly buffer tensorInB { float inB[]; };
layout (binding = 2) writeonly buffer tensorOut { float out_[]; };

layout (push_constant) uniform parameter {
    int M;
    int N;
    int K;
    int inAStride;
    int inBStride;
    int outStride;
    uint inAOff;
    uint inBOff;
    uint outOff;
} pcs;

shared float bufA[BM * (BK+1)];
shared float bufB[BN * (BK+1)];

void main() {
    const int ir = int(gl_WorkGroupID.x);
    const int ic = int(gl_WorkGroupID.y);

    const int rstride = BM / TM;

    const int lr = int(gl_WorkGroupID.x % rstride);
    const int lc = int(gl_WorkGroupID.x / rstride);

    const int loadr = int(gl_WorkGroupID.x % BK);
    const int loadc = int(gl_WorkGroupID.x / BK);

    const int loadstride = int(gl_WorkGroupSize.x);

    int posA = ir * BM * pcs.inAStride;
    int posB = ic * BN * pcs.inBStride;

    float sums[TM * TN];
    float cacheA[TM];
    float cacheB[TN];

    [[unroll]] for (int i = 0; i < TM*TN; i++) {
        sums[i] = 0.0f;
    }

    [[unroll]] for (int block = 0; block < pcs.K; block += BK) {
        [[unroll]] for (int l = 0; l < BM * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            bufA[(loadc + lc) * (BK+1) + loadr + lr] = inA[posA + (loadc + lc) * pcs.inAStride + loadr + lr + pcs.inAOff];
        }
        [[unroll]] for (int l = 0; l < BN * BK; l += loadstride) {
            const int lr = l % BK;
            const int lc = l / BK;
            bufB[(loadc + lc) * (BK+1) + loadr + lr] = inB[posB + (loadc + lc) * pcs.inBStride + loadr + lr + pcs.inBOff];
        }

        barrier();
        memoryBarrierShared();

        posA += BK;
        posB += BK;

        [[unroll]] for (int i = 0; i < BK; i++) {
            // Load from shared into cache
            [[unroll]] for (int j = 0; j < BM; j++) {
                cacheA[j] = bufA[(lr + j*rstride) * (BK+1) + i];
            }
            [[unroll]] for (int j = 0; j < TN; j++) {
                cacheB[j] = bufB[(lc * TN + j) * (BK+1) + i];
            }

            [[unroll]] for (int cc = 0; cc < TN; cc++) {
                [[unroll]] for (int cr = 0; cr < TM; cr++) {
                    sums[cc * TM + cr] += cacheA[cr] * cacheB[cc];
                }
            }
        }

        barrier();
    }

    const int dr = ir * BM + lr;
    const int dc = ic * BN + lc * TN;

    [[unroll]] for (int cc = 0; cc < TN; cc++) {
        [[unroll]] for (int cr = 0; cr < TM; cr++) {
            out_[(dc + cc) * pcs.outStride + dr + cr*rstride + pcs.outOff] = sums[cc * TM + cr];
        }
    }
}
);

void ggml_vk_mul_mat_f32(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA, uint32_t inAOff,
                         const std::shared_ptr<kp::Tensor>& inB, uint32_t inBOff,
                         const std::shared_ptr<kp::Tensor>& out, uint32_t outOff,
                         int64_t ne00, int64_t ne01, int64_t ne02, uint64_t ne03,
                         int64_t ne10, int64_t ne11,
                         int nb2, int nb3) {
    const static auto spirv = glsl_compile_source(program_source_head+program_mul_mat_f32, __func__);

    struct PushConstants {
        int32_t M, N, K, inAStride, inBStride, outStride;
        uint32_t inAOff, inBOff, outOff;
    } pushConsts {
        (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01,
        inAOff, inBOff, outOff
    };

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            auto off = i02*nb2 + i03*nb3;
            pushConsts.inAOff = inAOff + off;
            pushConsts.inBOff = inBOff + off;
            pushConsts.outOff = outOff + off;
            seq.record<kp::OpAlgoDispatch>(mgr.algorithm<float, PushConstants>({inA, inB, out}, spirv, {uint32_t(ne01/128), uint32_t(ne11/128)}, {}, {pushConsts}));
        }
    }
}


void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf) {
    printf("%s: evaluating graph\n", __func__);

    const int n_seq = gf->n_threads;

    std::vector<std::shared_ptr<kp::Sequence>> sequences(n_seq);

    for (auto& sequence : sequences) {
        sequence = mgr.sequence();
    }
    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int n_nodes_per_seq = (gf->n_nodes + n_seq - 1) / n_seq;

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

            const static std::shared_ptr<kp::Tensor> nullTensor = nullptr;
            const std::shared_ptr<kp::Tensor>& id_src0 = src0 ? ggml_vk_get_tensor(ctx, src0) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_src1 = src1 ? ggml_vk_get_tensor(ctx, src1) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_dst  = dst  ? ggml_vk_get_tensor(ctx, dst)  : nullTensor;

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
                        ggml_vk_abmath<'+'>(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ggml_nelements(dst));
                    } break;
                case GGML_OP_MUL:
                    {
                        if (ggml_nelements(src1) == ne10) {
                            // src1 is a row
                            ggml_vk_abmath<'*', true>(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ggml_nelements(dst), ne00);
                        } else {
                            ggml_vk_abmath<'*'>(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ggml_nelements(dst));
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
                case GGML_OP_SOFT_MAX:
                    {
                        ggml_vk_soft_max(seq, id_src0, offs_src0, id_dst, offs_dst, ne00, ne01, ne02, ne03);
                    } break;
                case GGML_OP_DIAG_MASK_INF:
                    {
                        ggml_vk_diag_mask_inf(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ne00, ne01, ne02);
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        if (src0->type == GGML_TYPE_F32
                         && src1->type == GGML_TYPE_F32) {
                            ggml_vk_mul_mat_f32(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ne00, ne01, ne02, ne03, ne10, ne11, nb2, nb3);
                            break;
                        } else if (src0->type == GGML_TYPE_F16
                                && src1->type == GGML_TYPE_F32) {
                            ggml_vk_mul_mat_f16(seq, id_src0, offs_src0, id_src1, offs_src1, id_dst, offs_dst, ne00, ne01, ne02, ne03, ne10, ne11, nb10, nb11, nb12, nb13, nb2, nb3);
                            break;
                        }
                    }
                default:
                    fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                    //GGML_ASSERT(false);
            }
        }

        // Evaluate sequence
        seq.evalAsync();
    }

    // Wait for all sequences to finish
    for (auto& sequence : sequences) {
        if (sequence->isRunning())
            sequence->evalAwait();
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
