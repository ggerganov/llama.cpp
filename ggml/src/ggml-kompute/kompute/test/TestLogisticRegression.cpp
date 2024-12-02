// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "kompute/Kompute.hpp"
#include "kompute/logger/Logger.hpp"

#include "test_logistic_regression_shader.hpp"

TEST(TestLogisticRegression, TestMainLogisticRegression)
{

    uint32_t ITERATIONS = 100;
    float learningRate = 0.1;

    {
        kp::Manager mgr;

        std::shared_ptr<kp::TensorT<float>> xI = mgr.tensor({ 0, 1, 1, 1, 1 });
        std::shared_ptr<kp::TensorT<float>> xJ = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> y = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> wIn = mgr.tensor({ 0.001, 0.001 });
        std::shared_ptr<kp::TensorT<float>> wOutI =
          mgr.tensor({ 0, 0, 0, 0, 0 });
        std::shared_ptr<kp::TensorT<float>> wOutJ =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> bIn = mgr.tensor({ 0 });
        std::shared_ptr<kp::TensorT<float>> bOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> lOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::vector<std::shared_ptr<kp::Tensor>> params = { xI,  xJ,    y,
                                                            wIn, wOutI, wOutJ,
                                                            bIn, bOut,  lOut };

        mgr.sequence()->eval<kp::OpTensorSyncDevice>(params);

        std::vector<uint32_t> spirv2{ 0x1, 0x2 };

        std::vector<uint32_t> spirv(
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.begin(),
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.end());

        std::shared_ptr<kp::Algorithm> algorithm = mgr.algorithm(
          params, spirv, kp::Workgroup({ 5 }), std::vector<float>({ 5.0 }));

        std::shared_ptr<kp::Sequence> sq =
          mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({ wIn, bIn })
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpTensorSyncLocal>({ wOutI, wOutJ, bOut, lOut });

        // Iterate across all expected iterations
        for (size_t i = 0; i < ITERATIONS; i++) {
            sq->eval();

            for (size_t j = 0; j < bOut->size(); j++) {
                wIn->data()[0] -= learningRate * wOutI->data()[j];
                wIn->data()[1] -= learningRate * wOutJ->data()[j];
                bIn->data()[0] -= learningRate * bOut->data()[j];
            }
        }

        // Based on the inputs the outputs should be at least:
        // * wi < 0.01
        // * wj > 1.0
        // * b < 0
        // TODO: Add EXPECT_DOUBLE_EQ instead
        EXPECT_LT(wIn->data()[0], 0.01);
        EXPECT_GT(wIn->data()[1], 1.0);
        EXPECT_LT(bIn->data()[0], 0.0);

        KP_LOG_WARN("Result wIn i: {}, wIn j: {}, bIn: {}",
                    wIn->data()[0],
                    wIn->data()[1],
                    bIn->data()[0]);
    }
}

TEST(TestLogisticRegression, TestMainLogisticRegressionManualCopy)
{

    uint32_t ITERATIONS = 100;
    float learningRate = 0.1;

    {
        kp::Manager mgr;

        std::shared_ptr<kp::TensorT<float>> xI = mgr.tensor({ 0, 1, 1, 1, 1 });
        std::shared_ptr<kp::TensorT<float>> xJ = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> y = mgr.tensor({ 0, 0, 0, 1, 1 });

        std::shared_ptr<kp::TensorT<float>> wIn =
          mgr.tensor({ 0.001, 0.001 }, kp::Tensor::TensorTypes::eHost);
        std::shared_ptr<kp::TensorT<float>> wOutI =
          mgr.tensor({ 0, 0, 0, 0, 0 });
        std::shared_ptr<kp::TensorT<float>> wOutJ =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> bIn =
          mgr.tensor({ 0 }, kp::Tensor::TensorTypes::eHost);
        std::shared_ptr<kp::TensorT<float>> bOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::shared_ptr<kp::TensorT<float>> lOut =
          mgr.tensor({ 0, 0, 0, 0, 0 });

        std::vector<std::shared_ptr<kp::Tensor>> params = { xI,  xJ,    y,
                                                            wIn, wOutI, wOutJ,
                                                            bIn, bOut,  lOut };

        mgr.sequence()->record<kp::OpTensorSyncDevice>(params)->eval();

        std::vector<uint32_t> spirv(
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.begin(),
          kp::TEST_LOGISTIC_REGRESSION_SHADER_COMP_SPV.end());

        std::shared_ptr<kp::Algorithm> algorithm = mgr.algorithm(
          params, spirv, kp::Workgroup(), std::vector<float>({ 5.0 }));

        std::shared_ptr<kp::Sequence> sq =
          mgr.sequence()
            ->record<kp::OpTensorSyncDevice>({ wIn, bIn })
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->record<kp::OpTensorSyncLocal>({ wOutI, wOutJ, bOut, lOut });

        // Iterate across all expected iterations
        for (size_t i = 0; i < ITERATIONS; i++) {

            sq->eval();

            for (size_t j = 0; j < bOut->size(); j++) {
                wIn->data()[0] -= learningRate * wOutI->data()[j];
                wIn->data()[1] -= learningRate * wOutJ->data()[j];
                bIn->data()[0] -= learningRate * bOut->data()[j];
            }
        }

        // Based on the inputs the outputs should be at least:
        // * wi < 0.01
        // * wj > 1.0
        // * b < 0
        // TODO: Add EXPECT_DOUBLE_EQ instead
        EXPECT_LT(wIn->data()[0], 0.01);
        EXPECT_GT(wIn->data()[1], 1.0);
        EXPECT_LT(bIn->data()[0], 0.0);

        KP_LOG_WARN("Result wIn i: {}, wIn j: {}, bIn: {}",
                    wIn->data()[0],
                    wIn->data()[1],
                    bIn->data()[0]);
    }
}
