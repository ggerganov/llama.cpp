// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include <chrono>

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"
#include "shaders/Utils.hpp"

TEST(TestAsyncOperations, TestManagerParallelExecution)
{
    // This test is built for NVIDIA 1650. It assumes:
    // * Queue family 0 and 2 have compute capabilities
    // * GPU is able to process parallel shader code across different families
    uint32_t size = 10;

    uint32_t numParallel = 2;

    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer b { float pb[]; };

        shared uint sharedTotal[1];

        void main() {
            uint index = gl_GlobalInvocationID.x;

            sharedTotal[0] = 0;

            for (int i = 0; i < 100000000; i++)
            {
                atomicAdd(sharedTotal[0], 1);
            }

            pb[index] = sharedTotal[0];
        }
    )");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::vector<float> data(size, 0.0);
    std::vector<float> resultSync(size, 100000000);
    std::vector<float> resultAsync(size, 100000000);

    kp::Manager mgr;

    std::shared_ptr<kp::Sequence> sq = mgr.sequence();

    std::vector<std::shared_ptr<kp::Tensor>> inputsSyncB;
    std::vector<std::shared_ptr<kp::Algorithm>> algorithms;

    for (uint32_t i = 0; i < numParallel; i++) {
        inputsSyncB.push_back(mgr.tensor(data));
        algorithms.push_back(mgr.algorithm({ inputsSyncB[i] }, spirv));
    }

    sq->eval<kp::OpTensorSyncDevice>(inputsSyncB);

    mgr.sequence()->eval<kp::OpTensorSyncDevice>(inputsSyncB);

    auto startSync = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < numParallel; i++) {
        sq->eval<kp::OpAlgoDispatch>(algorithms[i]);
    }

    auto endSync = std::chrono::high_resolution_clock::now();
    auto durationSync =
      std::chrono::duration_cast<std::chrono::microseconds>(endSync - startSync)
        .count();

    sq->eval<kp::OpTensorSyncLocal>(inputsSyncB);

    for (uint32_t i = 0; i < numParallel; i++) {
        EXPECT_EQ(inputsSyncB[i]->vector<float>(), resultSync);
    }

    kp::Manager mgrAsync(0, { 0, 2 });

    std::vector<std::shared_ptr<kp::Tensor>> inputsAsyncB;

    std::vector<std::shared_ptr<kp::Algorithm>> algosAsync;

    for (uint32_t i = 0; i < numParallel; i++) {
        inputsAsyncB.push_back(mgr.tensor(data));
        algosAsync.push_back(mgr.algorithm({ inputsAsyncB[i] }, spirv));
    }

    std::vector<std::shared_ptr<kp::Sequence>> sqs;

    for (uint32_t i = 0; i < numParallel; i++) {
        sqs.push_back(mgrAsync.sequence(i));
    }

    auto startAsync = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < numParallel; i++) {
        sqs[i]->evalAsync<kp::OpAlgoDispatch>(algosAsync[i]);
    }

    for (uint32_t i = 0; i < numParallel; i++) {
        sqs[i]->evalAwait();
    }

    auto endAsync = std::chrono::high_resolution_clock::now();
    auto durationAsync = std::chrono::duration_cast<std::chrono::microseconds>(
                           endAsync - startAsync)
                           .count();

    sq->eval<kp::OpTensorSyncLocal>({ inputsAsyncB });

    for (uint32_t i = 0; i < numParallel; i++) {
        EXPECT_EQ((inputsAsyncB[i]->vector<float>()), resultAsync);
    }

    // The speedup should be at least 40%
    EXPECT_LT(durationAsync, durationSync * 0.6);
}

TEST(TestAsyncOperations, TestManagerAsyncExecution)
{
    uint32_t size = 10;

    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer b { float pb[]; };

        shared uint sharedTotal[1];

        void main() {
            uint index = gl_GlobalInvocationID.x;

            sharedTotal[0] = 0;

            for (int i = 0; i < 100000000; i++)
            {
                atomicAdd(sharedTotal[0], 1);
            }

            pb[index] = sharedTotal[0];
        }
    )");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::vector<float> data(size, 0.0);
    std::vector<float> resultAsync(size, 100000000);

    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor(data);
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor(data);

    std::shared_ptr<kp::Sequence> sq1 = mgr.sequence();
    std::shared_ptr<kp::Sequence> sq2 = mgr.sequence();

    sq1->eval<kp::OpTensorSyncLocal>({ tensorA, tensorB });

    std::shared_ptr<kp::Algorithm> algo1 = mgr.algorithm({ tensorA }, spirv);
    std::shared_ptr<kp::Algorithm> algo2 = mgr.algorithm({ tensorB }, spirv);

    // AMD Drivers in Windows may see an error in this line due to timeout.
    // In order to fix this, it requires a change on Windows registries.
    // More details on this can be found here:
    // https://docs.substance3d.com/spdoc/gpu-drivers-crash-with-long-computations-128745489.html
    // Context on solution discussed in github:
    // https://github.com/KomputeProject/kompute/issues/196#issuecomment-808866505
    sq1->evalAsync<kp::OpAlgoDispatch>(algo1);
    sq2->evalAsync<kp::OpAlgoDispatch>(algo2);

    sq1->evalAwait();
    sq2->evalAwait();

    sq1->evalAsync<kp::OpTensorSyncLocal>({ tensorA, tensorB });
    sq1->evalAwait();

    EXPECT_EQ(tensorA->vector(), resultAsync);
    EXPECT_EQ(tensorB->vector(), resultAsync);
}

TEST(TestAsyncOperations, TestManagerAsyncExecutionTimeout)
{
    uint32_t size = 10;

    std::string shader(R"(
        #version 450

        layout (local_size_x = 1) in;

        layout(set = 0, binding = 0) buffer b { float pb[]; };

        shared uint sharedTotal[1];

        void main() {
            uint index = gl_GlobalInvocationID.x;

            sharedTotal[0] = 0;

            for (int i = 0; i < 100000000; i++)
            {
                atomicAdd(sharedTotal[0], 1);
            }

            pb[index] = sharedTotal[0];
        }
    )");

    std::vector<uint32_t> spirv = compileSource(shader);

    std::vector<float> data(size, 0.0);
    std::vector<float> resultAsync(size, 100000000);

    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorA = mgr.tensor(data);
    std::shared_ptr<kp::TensorT<float>> tensorB = mgr.tensor(data);

    std::shared_ptr<kp::Sequence> sq1 = mgr.sequence();
    std::shared_ptr<kp::Sequence> sq2 = mgr.sequence();

    sq1->eval<kp::OpTensorSyncLocal>({ tensorA, tensorB });

    std::shared_ptr<kp::Algorithm> algo1 = mgr.algorithm({ tensorA }, spirv);
    std::shared_ptr<kp::Algorithm> algo2 = mgr.algorithm({ tensorB }, spirv);

    auto startSync = std::chrono::high_resolution_clock::now();

    // AMD Drivers in Windows may see an error in this line due to timeout.
    // In order to fix this, it requires a change on Windows registries.
    // More details on this can be found here:
    // https://docs.substance3d.com/spdoc/gpu-drivers-crash-with-long-computations-128745489.html
    // Context on solution discussed in github:
    // https://github.com/KomputeProject/kompute/issues/196#issuecomment-808866505
    sq1->evalAsync<kp::OpAlgoDispatch>(algo1);
    sq2->evalAsync<kp::OpAlgoDispatch>(algo2);

    sq1->evalAwait(1);
    sq2->evalAwait(1);

    auto endSync = std::chrono::high_resolution_clock::now();
    auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(endSync - startSync)
        .count();

    // The time should several orders of magnitude smaller (in this 100k instead
    // of 1m ns)
    EXPECT_LT(duration, 100000);

    sq1->evalAsync<kp::OpTensorSyncLocal>({ tensorA, tensorB });
    sq1->evalAwait();

    EXPECT_EQ(tensorA->vector(), resultAsync);
    EXPECT_EQ(tensorB->vector(), resultAsync);
}
