// SPDX-License-Identifier: Apache-2.0

#include "kompute/operations/OpTensorFill.hpp"
#include "kompute/Tensor.hpp"

namespace kp {

OpTensorFill::OpTensorFill(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
    KP_LOG_DEBUG("Kompute OpTensorFill constructor with params");

    if (tensors.size() < 1) {
        throw std::runtime_error(
          "Kompute OpTensorFill called with less than 1 tensor");
    }

    this->mTensors = tensors;
}

OpTensorFill::~OpTensorFill()
{
    KP_LOG_DEBUG("Kompute OpTensorFill destructor started");
}

void
OpTensorFill::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorFill record called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        this->mTensors[i]->recordFill(commandBuffer, 0);
    }
}

void
OpTensorFill::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorFill preEval called");
}

void
OpTensorFill::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorFill postEval called");
}

}
