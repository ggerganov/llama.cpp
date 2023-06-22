// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Algorithm.hpp"
#include "kompute/Core.hpp"
#include "kompute/Tensor.hpp"

namespace kp {

/**
 *  Base Operation which provides the high level interface that Kompute
 *  operations implement in order to perform a set of actions in the GPU.
 *
 *  Operations can perform actions on tensors, and optionally can also own an
 *  Algorithm with respective parameters. kp::Operations with kp::Algorithms
 *  would inherit from kp::OpBaseAlgo.
 */
class OpBase
{
  public:
    /**
     * Default destructor for OpBase class. This OpBase destructor class should
     * always be called to destroy and free owned resources unless it is
     * intended to destroy the resources in the parent class.
     */
    virtual ~OpBase() { KP_LOG_DEBUG("Kompute OpBase destructor started"); }

    /**
     * The record function is intended to only send a record command or run
     * commands that are expected to record operations that are to be submitted
     * as a batch into the GPU.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void record(const vk::CommandBuffer& commandBuffer) = 0;

    /**
     * Pre eval is called before the Sequence has called eval and submitted the
     * commands to the GPU for processing, and can be used to perform any
     * per-eval setup steps required as the computation iteration begins. It's
     * worth noting that there are situations where eval can be called multiple
     * times, so the resources that are created should be idempotent in case
     * it's called multiple times in a row.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void preEval(const vk::CommandBuffer& commandBuffer) = 0;

    /**
     * Post eval is called after the Sequence has called eval and submitted the
     * commands to the GPU for processing, and can be used to perform any
     * tear-down steps required as the computation iteration finishes. It's
     * worth noting that there are situations where eval can be called multiple
     * times, so the resources that are destroyed should not require a re-init
     * unless explicitly provided by the user.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void postEval(const vk::CommandBuffer& commandBuffer) = 0;
};

} // End namespace kp
