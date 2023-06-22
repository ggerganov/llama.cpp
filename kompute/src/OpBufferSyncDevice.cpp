// SPDX-License-Identifier: Apache-2.0

/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

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
