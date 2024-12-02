
#include <iostream>
#include <memory>
#include <vector>

#include "my_shader.hpp"
#include <kompute/Kompute.hpp>

int
main()
{
    kp::Manager mgr;

    std::shared_ptr<kp::TensorT<float>> tensorInA =
      mgr.tensor({ 2.0, 4.0, 6.0 });
    std::shared_ptr<kp::TensorT<float>> tensorInB =
      mgr.tensor({ 0.0, 1.0, 2.0 });
    std::shared_ptr<kp::TensorT<float>> tensorOut =
      mgr.tensor({ 0.0, 0.0, 0.0 });

    const std::vector<std::shared_ptr<kp::Tensor>> params = { tensorInA,
                                                              tensorInB,
                                                              tensorOut };

    const std::vector<uint32_t> shader = std::vector<uint32_t>(
      shader::MY_SHADER_COMP_SPV.begin(), shader::MY_SHADER_COMP_SPV.end());
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, shader);

    mgr.sequence()
      ->record<kp::OpTensorSyncDevice>(params)
      ->record<kp::OpAlgoDispatch>(algo)
      ->record<kp::OpTensorSyncLocal>(params)
      ->eval();

    // prints "Output {  0  4  12  }"
    std::cout << "Output: {  ";
    for (const float& elem : tensorOut->vector()) {
        std::cout << elem << "  ";
    }
    std::cout << "}" << std::endl;

    if (tensorOut->vector() != std::vector<float>{ 0, 4, 12 }) {
        throw std::runtime_error("Result does not match");
    }
}
