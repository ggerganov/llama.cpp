// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "shaders/Utils.hpp"

TEST(TestSequence, SequenceDestructorViaManager)
{
    std::shared_ptr<kp::Sequence> sq = nullptr;

    {
        kp::Manager mgr;

        sq = mgr.sequence();

        EXPECT_TRUE(sq->isInit());
    }

    EXPECT_FALSE(sq->isInit());
}

TEST(TestSequence, SequenceDestructorOutsideManagerExplicit)
{
    std::shared_ptr<kp::Sequence> sq = nullptr;

    {
        kp::Manager mgr;

        sq = mgr.sequence();

        EXPECT_TRUE(sq->isInit());

        sq->destroy();

        EXPECT_FALSE(sq->isInit());
    }

    EXPECT_FALSE(sq->isInit());
}

TEST(TestSequence, SequenceDestructorOutsideManagerImplicit)
{
    kp::Manager mgr;

    std::weak_ptr<kp::Sequence> sqWeak;

    {
        std::shared_ptr<kp::Sequence> sq = mgr.sequence();

        sqWeak = sq;

        EXPECT_TRUE(sq->isInit());
    }

    EXPECT_FALSE(sqWeak.lock());
}

TEST(TestSequence, RerecordSequence)
{
    kp::Manager mgr;

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 1, 2, 3 });
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor({ 2, 2, 2 });
    std::shared_ptr<kp::TensorT<float>> tensorOut = mgr.tensor({ 0, 0, 0 });

    sq->eval<kp::OpTensorSyncDevice>({ tensorA, tensorB, tensorOut });

    std::vector<uint32_t> spirv = compileSource(R"(
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer bina { float tina[]; };
        layout(set = 0, binding = 1) buffer binb { float tinb[]; };
        layout(set = 0, binding = 2) buffer bout { float tout[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            tout[index] = tina[index] * tinb[index];
        }
    )");

    std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({ tensorA, tensorB, tensorOut }, spirv);

    sq->record<kp::OpAlgoDispatch>(algo)->record<kp::OpTensorSyncLocal>(
      { tensorA, tensorB, tensorOut });

    sq->eval();

    EXPECT_EQ(tensorOut->vector(), std::vector<float>({ 2, 4, 6 }));

    algo->rebuild({ tensorOut, tensorA, tensorB }, spirv);

    // Refresh and trigger a rerecord
    sq->rerecord();
    sq->eval();

    EXPECT_EQ(tensorB->vector(), std::vector<float>({ 2, 8, 18 }));
}

TEST(TestSequence, SequenceTimestamps)
{
    kp::Manager mgr;

    std::shared_ptr<kp::Tensor> tensorA = mgr.tensor({ 0, 0, 0 });

    std::string shader(R"(
      #version 450
      layout (local_size_x = 1) in;
      layout(set = 0, binding = 0) buffer a { float pa[]; };
      void main() {
          uint index = gl_GlobalInvocationID.x;
          pa[index] = pa[index] + 1;
      })");

    std::vector<uint32_t> spirv = compileSource(shader);

    auto seq = mgr.sequence(0, 100); // 100 timestamps
    seq->record<kp::OpTensorSyncDevice>({ tensorA })
      ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
      ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
      ->record<kp::OpAlgoDispatch>(mgr.algorithm({ tensorA }, spirv))
      ->record<kp::OpTensorSyncLocal>({ tensorA })
      ->eval();
    const std::vector<uint64_t> timestamps = seq->getTimestamps();

    EXPECT_EQ(timestamps.size(),
              6); // 1 timestamp at start + 1 after each operation
}

TEST(TestSequence, UtilsClearRecordingRunning)
{
    kp::Manager mgr;

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 1, 2, 3 });
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor({ 2, 2, 2 });
    std::shared_ptr<kp::TensorT<float>> tensorOut = mgr.tensor({ 0, 0, 0 });

    sq->eval<kp::OpTensorSyncDevice>({ tensorA, tensorB, tensorOut });

    std::vector<uint32_t> spirv = compileSource(R"(
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer bina { float tina[]; };
        layout(set = 0, binding = 1) buffer binb { float tinb[]; };
        layout(set = 0, binding = 2) buffer bout { float tout[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            tout[index] = tina[index] * tinb[index];
        }
    )");

    std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({ tensorA, tensorB, tensorOut }, spirv);

    sq->record<kp::OpAlgoDispatch>(algo)->record<kp::OpTensorSyncLocal>(
      { tensorA, tensorB, tensorOut });

    EXPECT_TRUE(sq->isRecording());

    // Running clear to confirm it clears
    sq->clear();

    EXPECT_FALSE(sq->isRecording());

    sq->evalAsync();

    EXPECT_TRUE(sq->isRunning());

    sq->evalAwait();

    EXPECT_FALSE(sq->isRunning());

    EXPECT_EQ(tensorOut->vector(), std::vector<float>({ 2, 4, 6 }));
}

TEST(TestSequence, CorrectSequenceRunningError)
{
    kp::Manager mgr;

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor({ 1, 2, 3 });
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor({ 2, 2, 2 });
    std::shared_ptr<kp::TensorT<float>> tensorOut = mgr.tensor({ 0, 0, 0 });

    sq->eval<kp::OpTensorSyncDevice>({ tensorA, tensorB, tensorOut });

    std::vector<uint32_t> spirv = compileSource(R"(
        #version 450

        layout (local_size_x = 1) in;

        // The input tensors bind index is relative to index in parameter passed
        layout(set = 0, binding = 0) buffer bina { float tina[]; };
        layout(set = 0, binding = 1) buffer binb { float tinb[]; };
        layout(set = 0, binding = 2) buffer bout { float tout[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            tout[index] = tina[index] * tinb[index];
        }
    )");

    std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm({ tensorA, tensorB, tensorOut }, spirv);

    sq->record<kp::OpAlgoDispatch>(algo)->record<kp::OpTensorSyncLocal>(
      { tensorA, tensorB, tensorOut });

    EXPECT_TRUE(sq->isRecording());

    sq->evalAsync();

    EXPECT_TRUE(sq->isRunning());

    // Sequence should throw when running
    EXPECT_ANY_THROW(sq->begin());
    EXPECT_ANY_THROW(sq->end());
    EXPECT_ANY_THROW(sq->evalAsync());

    // Errors should still not get into inconsystent state
    sq->evalAwait();

    // Sequence should not throw when finished
    EXPECT_NO_THROW(sq->evalAwait());
    EXPECT_NO_THROW(sq->evalAwait(10));

    EXPECT_FALSE(sq->isRunning());

    EXPECT_EQ(tensorOut->vector(), std::vector<float>({ 2, 4, 6 }));
}
