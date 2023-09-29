// SPDX-License-Identifier: Apache-2.0

/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

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
