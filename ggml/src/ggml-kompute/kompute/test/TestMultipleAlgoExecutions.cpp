// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "shaders/Utils.hpp"

TEST(TestMultipleAlgoExecutions, TestEndToEndFunctionality)
{

    kp::Manager mgr;

    // Default tensor constructor simplifies creation of float values
    auto tensorInA = mgr.tensor({ 2., 2., 2. });
    auto tensorInB = mgr.tensor({ 1., 2., 3. });
    // Explicit type constructor supports int, in32, double, float and int
    auto tensorOutA = mgr.tensorT<uint32_t>({ 0, 0, 0 });
    auto tensorOutB = mgr.tensorT<uint32_t>({ 0, 0, 0 });

    std::string shader = (R"(
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
        layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

        // Kompute supports push constants updated on dispatch
        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        // Kompute also supports spec constants on initalization
        layout(constant_id = 0) const float const_one = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_one * push_const.val );
        }
    )");

    std::vector<std::shared_ptr<kp::Tensor>> params = {
        tensorInA, tensorInB, tensorOutA, tensorOutB
    };

    kp::Workgroup workgroup({ 3, 1, 1 });
    std::vector<float> specConsts({ 2 });
    std::vector<float> pushConstsA({ 2.0 });
    std::vector<float> pushConstsB({ 3.0 });

    auto algorithm = mgr.algorithm(
      params, compileSource(shader), workgroup, specConsts, pushConstsA);

    // 3. Run operation with string shader synchronously
    mgr.sequence()
      ->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algorithm)
      ->eval()
      ->record<kp::OpAlgoDispatch>(algorithm, pushConstsB)
      ->eval();

    auto sq = mgr.sequence();
    sq->evalAsync<kp::OpTensorSyncLocal>(params);

    sq->evalAwait();

    EXPECT_EQ(tensorOutA->vector(), std::vector<uint32_t>({ 4, 8, 12 }));
    EXPECT_EQ(tensorOutB->vector(), std::vector<uint32_t>({ 10, 10, 10 }));
}

TEST(TestMultipleAlgoExecutions, SingleSequenceRecord)
{

    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    {
        // A sharedMemoryBarrier is required as the shader is not thread-safe:w
        std::shared_ptr<kp::OpMemoryBarrier> shaderBarrier{
            new kp::OpMemoryBarrier({ tensorA },
                                    vk::AccessFlagBits::eTransferRead,
                                    vk::AccessFlagBits::eShaderWrite,
                                    vk::PipelineStageFlagBits::eComputeShader,
                                    vk::PipelineStageFlagBits::eComputeShader)
        };

        mgr.sequence()
          ->record<kp::OpTensorSyncDevice>({ tensorA })
          ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
          ->record(shaderBarrier)
          ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
          ->record(shaderBarrier)
          ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
          ->record<kp::OpTensorSyncLocal>({ tensorA })
          ->eval();
    }

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 3, 3, 3 }));
}

TEST(TestMultipleAlgoExecutions, MultipleCmdBufRecords)
{
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::shared_ptr<kp::Algorithm> algorithm =
      mgr.algorithm({ tensorA }, spirv);

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    mgr.sequence()->record<kp::OpTensorSyncDevice>({ tensorA })->eval();

    mgr.sequence()->record<kp::OpAlgoDispatch>(algorithm)->eval();

    mgr.sequence()->record<kp::OpAlgoDispatch>(algorithm)->eval();

    mgr.sequence()->record<kp::OpAlgoDispatch>(algorithm)->eval();

    mgr.sequence()->record<kp::OpTensorSyncLocal>({ tensorA })->eval();

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 3, 3, 3 }));
}

TEST(TestMultipleAlgoExecutions, MultipleSequences)
{

    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::shared_ptr<kp::Algorithm> algorithm =
      mgr.algorithm({ tensorA }, spirv);

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    sq->record<kp::OpTensorSyncDevice>({ tensorA })->eval();

    sq->record<kp::OpAlgoDispatch>(algorithm)->eval();

    sq->record<kp::OpAlgoDispatch>(algorithm)->eval();

    sq->record<kp::OpAlgoDispatch>(algorithm)->eval();

    sq->record<kp::OpTensorSyncLocal>({ tensorA })->eval();

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 3, 3, 3 }));
}

TEST(TestMultipleAlgoExecutions, SingleRecordMultipleEval)
{
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::shared_ptr<kp::Algorithm> algorithm =
      mgr.algorithm({ tensorA }, spirv);

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    sq->record<kp::OpTensorSyncDevice>({ tensorA })->eval();

    sq->record<kp::OpAlgoDispatch>(algorithm)->eval()->eval()->eval();

    sq->record<kp::OpTensorSyncLocal>({ tensorA })->eval();

    EXPECT_EQ(tensorA->vector(), std::vector<float>({ 3, 3, 3 }));
}

TEST(TestMultipleAlgoExecutions, TestAlgorithmUtilFunctions)
{

    kp::Manager mgr;

    // Default tensor constructor simplifies creation of float values
    auto tensorInA = mgr.tensor({ 2., 2., 2. });
    auto tensorInB = mgr.tensor({ 1., 2., 3. });
    // Explicit type constructor supports int, in32, double, float and int
    auto tensorOutA = mgr.tensorT<uint32_t>({ 0, 0, 0 });
    auto tensorOutB = mgr.tensorT<uint32_t>({ 0, 0, 0 });

    std::string shader = (R"(
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer buf_in_a { float in_a[]; };
        layout(set = 0, binding = 1) buffer buf_in_b { float in_b[]; };
        layout(set = 0, binding = 2) buffer buf_out_a { uint out_a[]; };
        layout(set = 0, binding = 3) buffer buf_out_b { uint out_b[]; };

        // Kompute supports push constants updated on dispatch
        layout(push_constant) uniform PushConstants {
            float val;
        } push_const;

        // Kompute also supports spec constants on initalization
        layout(constant_id = 0) const float const_one = 0;

        void main() {
            uint index = gl_GlobalInvocationID.x;
            out_a[index] += uint( in_a[index] * in_b[index] );
            out_b[index] += uint( const_one * push_const.val );
        }
    )");

    std::vector<std::shared_ptr<kp::Tensor>> params = {
        tensorInA, tensorInB, tensorOutA, tensorOutB
    };

    kp::Workgroup workgroup({ 3, 1, 1 });
    std::vector<float> specConsts({ 2 });
    std::vector<float> pushConsts({ 2.0 });

    auto algorithm = mgr.algorithm(
      params, compileSource(shader), workgroup, specConsts, pushConsts);

    EXPECT_EQ(algorithm->getWorkgroup(), workgroup);
    EXPECT_EQ(algorithm->getPushConstants<float>(), pushConsts);
    EXPECT_EQ(algorithm->getSpecializationConstants<float>(), specConsts);
}
