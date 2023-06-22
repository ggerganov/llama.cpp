// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Algorithm.hpp"
#include "kompute/Core.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/operations/OpBase.hpp"

namespace kp {

/**
 * Operation that provides a general abstraction that simplifies the use of
 * algorithm and parameter components which can be used with shaders.
 * It exposes the pipeline barrier functionality specifically for memory
 * barriers that can be configured through the respective source and destination
 * masks
 */
class OpMemoryBarrier : public OpBase
{
  public:
    /**
     * Constructor that stores tensors as well as memory barrier parameters to
     * be used to create a pipeline barrier on the respective primary or staging
     * tensor.
     *
     * @param tensors The tensors to apply the memory barriers on
     * @param srcAccessMask The kp::AccessFlagBits for the source access mask
     * @param dstAccessMask The kp::AccessFlagBits for the destination access
     * mask
     * @param srcStageMask The kp::PipelineStageFlagBits for the source stage
     * mask
     * @param dstStageMask The kp::PipelineStageFlagBits for the destination
     * stage mask
     * @param barrierOnPrimary Boolean to select primary or secondary buffers on
     * tensors
     */
    OpMemoryBarrier(const std::vector<std::shared_ptr<Tensor>>& tensors,
                    const vk::AccessFlagBits& srcAccessMask,
                    const vk::AccessFlagBits& dstAccessMask,
                    const vk::PipelineStageFlagBits& srcStageMask,
                    const vk::PipelineStageFlagBits& dstStageMask,
                    bool barrierOnPrimary = true);

    /**
     * Default destructor, which is in charge of destroying the reference to the
     * tensors and all the relevant access / stage masks created
     */
    virtual ~OpMemoryBarrier() override;

    /**
     * This records the memory barrier with the access and stage masks provided
     * across all relevant tensors.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void record(const vk::CommandBuffer& commandBuffer) override;

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
    const vk::AccessFlagBits mSrcAccessMask;
    const vk::AccessFlagBits mDstAccessMask;
    const vk::PipelineStageFlagBits mSrcStageMask;
    const vk::PipelineStageFlagBits mDstStageMask;
    const bool mBarrierOnPrimary;
    const std::vector<std::shared_ptr<Tensor>> mTensors;
};

} // End namespace kp
