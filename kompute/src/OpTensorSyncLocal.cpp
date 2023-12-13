// SPDX-License-Identifier: Apache-2.0

#include "kompute/Tensor.hpp"

#include "kompute/operations/OpTensorSyncLocal.hpp"

namespace kp {

OpTensorSyncLocal::OpTensorSyncLocal(
  const std::vector<std::shared_ptr<Tensor>>& tensors)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal constructor with params");

    if (tensors.size() < 1) {
        throw std::runtime_error(
          "Kompute OpTensorSyncLocal called with less than 1 tensor");
    }

    this->mTensors = tensors;
}

OpTensorSyncLocal::~OpTensorSyncLocal()
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal destructor started");
}

void
OpTensorSyncLocal::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal record called");

    for (size_t i = 0; i < this->mTensors.size(); i++) {
        if (this->mTensors[i]->tensorType() == Tensor::TensorTypes::eDevice) {

            this->mTensors[i]->recordPrimaryBufferMemoryBarrier(
              commandBuffer,
              vk::AccessFlagBits::eShaderWrite,
              vk::AccessFlagBits::eTransferRead,
              vk::PipelineStageFlagBits::eComputeShader,
              vk::PipelineStageFlagBits::eTransfer);

            this->mTensors[i]->recordCopyFromDeviceToStaging(commandBuffer);

            this->mTensors[i]->recordPrimaryBufferMemoryBarrier(
              commandBuffer,
              vk::AccessFlagBits::eTransferWrite,
              vk::AccessFlagBits::eHostRead,
              vk::PipelineStageFlagBits::eTransfer,
              vk::PipelineStageFlagBits::eHost);
        }
    }
}

void
OpTensorSyncLocal::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal preEval called");
}

void
OpTensorSyncLocal::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpTensorSyncLocal postEval called");

    KP_LOG_DEBUG("Kompute OpTensorSyncLocal mapping data into tensor local");
}

}
