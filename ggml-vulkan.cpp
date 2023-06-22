#include "ggml-vulkan.h"
#include "ggml.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <kompute/Kompute.hpp>

#ifndef __STDC_IEC_559__
#error Your C implementation is not IEC 559 compliant, which is required for proper Vulkan interop.
#endif

typedef ggml_fp16_t half;

#define MULTILINE_QUOTE(...) #__VA_ARGS__
#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x

#define QK4_0 32
#define QR4_0 2
#define QK4_1 32

typedef struct {
    half    d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    half d;
    half m;
    uint8_t qs[QK4_1 / 2];
} block_q4_1;


kp::Manager mgr;



std::vector<uint32_t> compileSource(const std::string& source) {
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


static const std::string program_source_head = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#define QK4_0 32
#define QR4_0 2
#define QK4_1 32
layout (local_size_x = 1) in;
)";


static const std::string program_dequantize_row_q4_0 =
        program_source_head+'\n'+MULTILINE_QUOTE(
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

    auto getVecBlockQ4_0D = [x, nb] () {
        std::vector<half> fres(nb);
        for (unsigned it = 0; it != nb; it++) {
            fres[it] = x[it].d;
        }
        return fres;
    };
    auto getVecBlockQ4_0QS = [x, nb] () {
        std::vector<uint8_t> fres(nb*(qk/2));
        for (unsigned x_it = 0; x_it != nb; x_it++) {
            for (unsigned qs_it = 0; qs_it != qk / 2; qs_it++) {
                fres[x_it * (qk / 2) + qs_it] = x[x_it].qs[qs_it];
            }
        }
        return fres;
    };

    const auto tensorBlockQ4_0D = mgr.tensorT<half>(getVecBlockQ4_0D());
    const auto tensorBlockQ4_0QS = mgr.tensorT<uint8_t>(getVecBlockQ4_0QS());
    const auto tensorY = mgr.tensor(std::vector<float>(y, y+y_size));

    mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({tensorBlockQ4_0D, tensorBlockQ4_0QS, tensorY})
            ->record<kp::OpAlgoDispatch>(mgr.algorithm({tensorBlockQ4_0D, tensorBlockQ4_0QS, tensorY}, spirv, {nb, qk/2, 0}))
            ->record<kp::OpTensorSyncLocal>({tensorY})
            ->eval();

    std::memcpy(y, tensorY->data(), tensorY->size()*sizeof(*y));
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
