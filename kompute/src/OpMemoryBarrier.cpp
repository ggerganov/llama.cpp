// SPDX-License-Identifier: Apache-2.0

/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

#include "kompute/operations/OpMemoryBarrier.hpp"

namespace kp {

OpMemoryBarrier::OpMemoryBarrier(
  const std::vector<std::shared_ptr<Tensor>>& tensors,
  const vk::AccessFlagBits& srcAccessMask,
  const vk::AccessFlagBits& dstAccessMask,
  const vk::PipelineStageFlagBits& srcStageMask,
  const vk::PipelineStageFlagBits& dstStageMask,
  bool barrierOnPrimary)
  : mSrcAccessMask(srcAccessMask)
  , mDstAccessMask(dstAccessMask)
  , mSrcStageMask(srcStageMask)
  , mDstStageMask(dstStageMask)
  , mBarrierOnPrimary(barrierOnPrimary)
  , mTensors(tensors)
{
    KP_LOG_DEBUG("Kompute OpMemoryBarrier constructor");
}

OpMemoryBarrier::~OpMemoryBarrier()
{
    KP_LOG_DEBUG("Kompute OpMemoryBarrier destructor started");
}

void
OpMemoryBarrier::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpMemoryBarrier record called");

    // Barrier to ensure the data is finished writing to buffer memory
    if (this->mBarrierOnPrimary) {
        for (const std::shared_ptr<Tensor>& tensor : this->mTensors) {
            tensor->recordPrimaryBufferMemoryBarrier(commandBuffer,
                                                     this->mSrcAccessMask,
                                                     this->mDstAccessMask,
                                                     this->mSrcStageMask,
                                                     this->mDstStageMask);
        }
    } else {
        for (const std::shared_ptr<Tensor>& tensor : this->mTensors) {
            tensor->recordStagingBufferMemoryBarrier(commandBuffer,
                                                     this->mSrcAccessMask,
                                                     this->mDstAccessMask,
                                                     this->mSrcStageMask,
                                                     this->mDstStageMask);
        }
    }
}

void
OpMemoryBarrier::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpMemoryBarrier preEval called");
}

void
OpMemoryBarrier::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpMemoryBarrier postSubmit called");
}

}
