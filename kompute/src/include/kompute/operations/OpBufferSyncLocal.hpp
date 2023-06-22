// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/operations/OpBase.hpp"

namespace kp {

class OpBufferSyncLocal : public OpBase
{
  public:
    OpBufferSyncLocal(
        vk::Buffer *primaryBuffer,
        vk::Buffer *stagingBuffer,
        vk::DeviceSize size);

    /**
     * Default destructor. This class does not manage memory so it won't be
     * expecting the parent to perform a release.
     */
    ~OpBufferSyncLocal() override;

    /**
     * For device buffers, it records the copy command for the buffer to copy
     * the data from its staging to device memory.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    void record(const vk::CommandBuffer& commandBuffer) override;

    /**
     * Does not perform any preEval commands.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void preEval(const vk::CommandBuffer& commandBuffer) override;

    /**
     * Does not perform any postEval commands.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void postEval(const vk::CommandBuffer& commandBuffer) override;

  private:
    vk::Buffer *mPrimaryBuffer;
    vk::Buffer *mStagingBuffer;
    vk::DeviceSize mSize;
};

} // End namespace kp
