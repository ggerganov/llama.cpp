
#include <iostream>
#include <memory>
#include <vector>

#include "kompute/Tensor.hpp"
#include "my_shader.hpp"
#include <kompute/Kompute.hpp>

int
main()
{
    uint32_t ITERATIONS = 100;
    float learningRate = 0.1;

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
      shader::MY_SHADER_COMP_SPV.begin(),
      shader::MY_SHADER_COMP_SPV.end());

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

    KP_LOG_WARN("Result wIn i: {}, wIn j: {}, bIn: {}",
                wIn->data()[0],
                wIn->data()[1],
                bIn->data()[0]);

    if (wIn->data()[0] > 0.01 ||
            wIn->data()[1] < 1.0 ||
            bIn->data()[0] > 0.0) {
        throw std::runtime_error("Result does not match");
    }
}
