// SPDX-License-Identifier: Apache-2.0

#include "kompute/operations/OpBufferSyncDevice.hpp"

namespace kp {

OpBufferSyncDevice::OpBufferSyncDevice(
        vk::Buffer *primaryBuffer,
        vk::Buffer *stagingBuffer,
        vk::DeviceSize size)
  : mPrimaryBuffer(primaryBuffer)
  , mStagingBuffer(stagingBuffer)
  , mSize(size)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncDevice constructor with params");
}

OpBufferSyncDevice::~OpBufferSyncDevice()
{
    KP_LOG_DEBUG("Kompute OpBufferSyncDevice destructor started");
}

void
OpBufferSyncDevice::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncDevice record called");
    vk::BufferCopy copyRegion(0, 0, mSize);
    commandBuffer.copyBuffer(*mStagingBuffer, *mPrimaryBuffer, copyRegion);
}

void
OpBufferSyncDevice::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncDevice preEval called");
}

void
OpBufferSyncDevice::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncDevice postEval called");
}

}
