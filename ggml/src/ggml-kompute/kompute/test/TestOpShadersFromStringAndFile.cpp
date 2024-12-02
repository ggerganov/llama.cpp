// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "shaders/Utils.hpp"
#include "test_op_custom_shader.hpp"
#include "test_shader.hpp"

TEST(TestShaderEndianness, ShaderRawDataFromConstructor)
{
    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer a { float pa[]; };
        layout(set = 0, binding = 1) buffer b { float pb[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            pb[index] = pa[index];
            pa[index] = index;
        }
    )");

    std::vector<uint32_t> spirv = compileSource(shader);
    std::vector<uint32_t> spirv2(kp::TEST_SHADER_COMP_SPV.begin(),
                                 kp::TEST_SHADER_COMP_SPV.end());
    EXPECT_EQ(spirv.size(), spirv2.size());
    for (size_t i = 0; i < spirv.size(); i++) {
        EXPECT_EQ(spirv[i], spirv2[i]);
    }
}

TEST(TestOpAlgoCreate, ShaderRawDataFromConstructor)
{
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 3, 4, 5 });
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer a { float pa[]; };
        layout(set = 0, binding = 1) buffer b { float pb[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            pb[index] = pa[index];
            pa[index] = index;
        }
    )");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::vector<std::shared_ptr<kp::Tensor>> params = { tensorA, tensorB };

    mgr.sequence()
      ->eval<kp::OpTensorSyncDevice>(params)
      ->eval<kp::OpAlgoDispatch>(mgr.algorithm(params, spirv))
      ->eval<kp::OpTensorSyncLocal>(params);

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 0, 1, 2 }));
    EXPECT_EQ(tensorB->vector(), std::vector<float>({ 3, 4, 5 }));
}

TEST(TestOpAlgoCreate, ShaderCompiledDataFromConstructor)
{
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 3, 4, 5 });
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor({ 0, 0, 0 });

    std::vector<uint32_t> spirv(kp::TEST_OP_CUSTOM_SHADER_COMP_SPV.begin(),
                                kp::TEST_OP_CUSTOM_SHADER_COMP_SPV.end());
    std::vector<std::shared_ptr<kp::Tensor>> params = { tensorA, tensorB };

    mgr.sequence()
      ->eval<kp::OpTensorSyncDevice>(params)
      ->eval<kp::OpAlgoDispatch>(mgr.algorithm(params, spirv))
      ->eval<kp::OpTensorSyncLocal>(params);

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 0, 1, 2 }));
    EXPECT_EQ(tensorB->vector(), std::vector<float>({ 3, 4, 5 }));
}

// TODO: Add support to read from file for shader
// TEST(TestOpAlgoCreate, ShaderCompiledDataFromFile)
//{
//    kp::Manager mgr;
//
//    std::shared_ptr<kp::TensorT<float>> tensorA{ new kp::Tensor({ 3, 4, 5 })
//    }; std::shared_ptr<kp::TensorT<float>> tensorB{ new kp::Tensor({ 0, 0, 0
//    }) }; mgr.rebuild({ tensorA, tensorB });
//
//    mgr.evalOpDefault<kp::OpAlgoCreate>(
//      { tensorA, tensorB },
//      "test/shaders/glsl/test_op_custom_shader.comp.spv");
//
//    mgr.evalOpDefault<kp::OpTensorSyncLocal>({ tensorA, tensorB });
//
//    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 0, 1, 2 }));
//    EXPECT_EQ(tensorB->vector(), std::vector<float>({ 3, 4, 5 }));
//}
