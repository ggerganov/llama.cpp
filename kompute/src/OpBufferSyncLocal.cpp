// SPDX-License-Identifier: Apache-2.0

/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

#include "kompute/operations/OpBufferSyncLocal.hpp"

namespace kp {

OpBufferSyncLocal::OpBufferSyncLocal(
        vk::Buffer *primaryBuffer,
        vk::Buffer *stagingBuffer,
        vk::DeviceSize size)
  : mPrimaryBuffer(primaryBuffer)
  , mStagingBuffer(stagingBuffer)
  , mSize(size)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncLocal constructor with params");
}

OpBufferSyncLocal::~OpBufferSyncLocal()
{
    KP_LOG_DEBUG("Kompute OpBufferSyncLocal destructor started");
}

void
OpBufferSyncLocal::record(const vk::CommandBuffer& commandBuffer)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncLocal record called");
    vk::BufferCopy copyRegion(0, 0, mSize);
    commandBuffer.copyBuffer(*mPrimaryBuffer, *mStagingBuffer, copyRegion);
}

void
OpBufferSyncLocal::preEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncLocal preEval called");
}

void
OpBufferSyncLocal::postEval(const vk::CommandBuffer& /*commandBuffer*/)
{
    KP_LOG_DEBUG("Kompute OpBufferSyncLocal postEval called");
}

}
