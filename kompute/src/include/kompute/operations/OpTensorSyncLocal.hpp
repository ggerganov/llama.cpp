// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Core.hpp"

#include "kompute/Tensor.hpp"

#include "kompute/operations/OpBase.hpp"

namespace kp {

/**
 * Operation that syncs tensor's local memory by mapping device data into the
 * local CPU memory. For TensorTypes::eDevice it will use a record operation
 * for the memory to be syncd into GPU memory which means that the operation
 * will be done in sync with GPU commands. For TensorTypes::eHost it will
 * only map the data into host memory which will happen during preEval before
 * the recorded commands are dispatched.
 */
class OpTensorSyncLocal : public OpBase
{
  public:
    /**
     * Default constructor with parameters that provides the core vulkan
     * resources and the tensors that will be used in the operation. The tensors
     * provided cannot be of type TensorTypes::eStorage.
     *
     * @param tensors Tensors that will be used to create in operation.
     */
    OpTensorSyncLocal(const std::vector<std::shared_ptr<Tensor>>& tensors);

    /**
     * Default destructor. This class does not manage memory so it won't be
     * expecting the parent to perform a release.
     */
    ~OpTensorSyncLocal() override;

    /**
     * For device tensors, it records the copy command for the tensor to copy
     * the data from its device to staging memory.
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
     * For host tensors it performs the map command from the host memory into
     * local memory.
     *
     * @param commandBuffer The command buffer to record the command into.
     */
    virtual void postEval(const vk::CommandBuffer& commandBuffer) override;

  private:
    // -------------- ALWAYS OWNED RESOURCES
    std::vector<std::shared_ptr<Tensor>> mTensors;
};

} // End namespace kp
