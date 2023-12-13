// SPDX-License-Identifier: Apache-2.0

#include "kompute/operations/OpTensorCopy.hpp"
#include "kompute/Tensor.hpp"

namespace kp {

OpTensorCopy::OpTensorCopy(const std::vector<std::shared_ptr<Tensor>>& tensors)
{
    KP_LOG_DEBUG("Kompute OpTensorCopy constructor with params");

    this->mTensors = tensors;

    if (this->mTensors.size() < 2) {
        throw std::runtime_error(
          "Kompute OpTensorCopy called with less than 2 tensor");
    }

    kp::Tensor::TensorDataTypes dataType = this->mTensors[0]->dataType();
    uint32_t size = this->mTensors[0]->size();
    for (const std::shared_ptr<Tensor>& tensor : tensors) {
        if (tensor->dataType() != dataType) {
            throw std::runtime_error(fmt::format(
              "Attempting to copy tensors of different types from {} to {}",
              Tensor::toString(dataType),
              Tensor::toString(tensor->dataType())));
        }
        if (tensor->size() != size) {
            throw std::runtime_error(fmt::format(
              "Attempting to copy tensors of different sizes from {} to {}",
              size,
              tensor->size()));
        }
    }
}

OpTensorCopy::~OpTensorCopy()
{
    KP_LOG_DEBUG("Kompute OpTensorCopy destructor started");
}

void
OpTensorCopy::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorCopy record called");

    // We iterate from the second tensor onwards and record a copy to all
    for (size_t i = 1; i < this->mTensors.size(); i++) {
        this->mTensors[i]->recordCopyFrom(commandBuffer, this->mTensors[0]);
    }
}

void
OpTensorCopy::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorCopy preEval called");
}

void
OpTensorCopy::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorCopy postEval called");

    // Do not copy on CPU side if source is storage tensor
    if (this->mTensors[0]->tensorType() == kp::Tensor::TensorTypes::eStorage)
    {
        KP_LOG_DEBUG("Kompute OpTensorCopy not copying tensor source given it's of eStorage type");
        return;
    }
    void* data = this->mTensors[0]->rawData();

    // Copy the data from the first tensor into all the tensors
    for (size_t i = 1; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == kp::Tensor::TensorTypes::eStorage) {
            KP_LOG_DEBUG("Kompute OpTensorCopy not copying to tensor dest given it's of eStorage type");
            continue;
        }
        this->mTensors[i]->setRawData(data);
    }
}

}
