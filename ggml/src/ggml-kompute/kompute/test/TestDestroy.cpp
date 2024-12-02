// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "shaders/Utils.hpp"

TEST(TestDestroy, TestDestroyTensorSingle)
{
    std::shared_ptr<kp::TensorT<float>> tensorA = nullptr;

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
        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            const std::vector<float> initialValues = { 0.0f, 0.0f, 0.0f };

            tensorA = mgr.tensor(initialValues);

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm({ tensorA }, spirv);

            // Sync values to and from device
            mgr.sequence()->eval<kp::OpTensorSyncDevice>(algo->getTensors());

            EXPECT_EQ(tensorA->vector(), initialValues);

            mgr.sequence()
              ->record<kp::OpAlgoDispatch>(algo)
              ->eval()
              ->eval<kp::OpTensorSyncLocal>(algo->getTensors());

            const std::vector<float> expectedFinalValues = { 1.0f, 1.0f, 1.0f };
            EXPECT_EQ(tensorA->vector(), expectedFinalValues);

            tensorA->destroy();
            EXPECT_FALSE(tensorA->isInit());
        }
        EXPECT_FALSE(tensorA->isInit());
    }
}

TEST(TestDestroy, TestDestroyTensorVector)
{
    std::shared_ptr<kp::TensorT<float>> tensorA = nullptr;
    std::shared_ptr<kp::TensorT<float>> tensorB = nullptr;

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      layout(set = 0, binding = 1) buffer b { float pb[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
          pb[index] = pb[index] + 2;
      })");
    std::vector<uint32_t> spirv = compileSource(shader);

    {
        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            tensorA = mgr.tensor({ 1, 1, 1 });
            tensorB = mgr.tensor({ 1, 1, 1 });

            std::shared_ptr<kp::Algorithm> algo =
              mgr.algorithm({ tensorA, tensorB }, spirv);

            mgr.sequence()
              ->record<kp::OpTensorSyncDevice>(algo->getTensors())
              ->record<kp::OpAlgoDispatch>(algo)
              ->record<kp::OpTensorSyncLocal>(algo->getTensors())
              ->eval();

            EXPECT_EQ(tensorA->vector(), std::vector<float>({ 2, 2, 2 }));
            EXPECT_EQ(tensorB->vector(), std::vector<float>({ 3, 3, 3 }));

            tensorA->destroy();
            tensorB->destroy();

            EXPECT_FALSE(tensorA->isInit());
            EXPECT_FALSE(tensorB->isInit());
        }
    }
}

TEST(TestDestroy, TestDestroySequenceSingle)
{
    std::shared_ptr<kp::TensorT<float>> tensorA = nullptr;

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
        std::shared_ptr<kp::Sequence> sq = nullptr;

        {
            kp::Manager mgr;

            tensorA = mgr.tensor({ 0, 0, 0 });

            sq =
              mgr.sequence()
                ->record<kp::OpTensorSyncDevice>({ tensorA })
                ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
                ->record<kp::OpTensorSyncLocal>({ tensorA })
                ->eval();

            sq->destroy();

            EXPECT_FALSE(sq->isInit());

            EXPECT_EQ(tensorA->vector(), std::vector<float>({ 1, 1, 1 }));
        }
    }
}
