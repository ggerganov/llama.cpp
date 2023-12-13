// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Core.hpp"

#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpBase.hpp"

namespace kp {

/**
 *  Container of operations that can be sent to GPU as batch
 */
class Sequence : public std::enable_shared_from_this<Sequence>
{
  public:
    /**
     * Main constructor for sequence which requires core vulkan components to
     * generate all dependent resources.
     *
     * @param physicalDevice Vulkan physical device
     * @param device Vulkan logical device
     * @param computeQueue Vulkan compute queue
     * @param queueIndex Vulkan compute queue index in device
     * @param totalTimestamps Maximum number of timestamps to allocate
     */
    Sequence(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
             std::shared_ptr<vk::Device> device,
             std::shared_ptr<vk::Queue> computeQueue,
             uint32_t queueIndex,
             uint32_t totalTimestamps = 0);
    /**
     * Destructor for sequence which is responsible for cleaning all subsequent
     * owned operations.
     */
    ~Sequence();

    /**
     * Record function for operation to be added to the GPU queue in batch. This
     * template requires classes to be derived from the OpBase class. This
     * function also requires the Sequence to be recording, otherwise it will
     * not be able to add the operation.
     *
     * @param op Object derived from kp::BaseOp that will be recoreded by the
     * sequence which will be used when the operation is evaluated.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    std::shared_ptr<Sequence> record(std::shared_ptr<OpBase> op);

    /**
     * Record function for operation to be added to the GPU queue in batch. This
     * template requires classes to be derived from the OpBase class. This
     * function also requires the Sequence to be recording, otherwise it will
     * not be able to add the operation.
     *
     * @param tensors Vector of tensors to use for the operation
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> record(
      std::vector<std::shared_ptr<Tensor>> tensors,
      TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(tensors, std::forward<TArgs>(params)...) };
        return this->record(op);
    }
    /**
     * Record function for operation to be added to the GPU queue in batch. This
     * template requires classes to be derived from the OpBase class. This
     * function also requires the Sequence to be recording, otherwise it will
     * not be able to add the operation.
     *
     * @param algorithm Algorithm to use for the record often used for OpAlgo
     * operations
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> record(std::shared_ptr<Algorithm> algorithm,
                                     TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(algorithm,
                                     std::forward<TArgs>(params)...) };
        return this->record(op);
    }

    /**
     * Eval sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job synchronously (with a barrier).
     *
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    std::shared_ptr<Sequence> eval();

    /**
     * Resets all the recorded and stored operations, records the operation
     * provided and submits into the gpu as a submit job synchronously (with a
     * barrier).
     *
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    std::shared_ptr<Sequence> eval(std::shared_ptr<OpBase> op);

    /**
     * Eval sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job with a barrier.
     *
     * @param tensors Vector of tensors to use for the operation
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> eval(std::vector<std::shared_ptr<Tensor>> tensors,
                                   TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(tensors, std::forward<TArgs>(params)...) };
        return this->eval(op);
    }

    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> eval(vk::Buffer *primaryBuffer,
                                   vk::Buffer *stagingBuffer,
                                   vk::DeviceSize size,
                                   TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(primaryBuffer, stagingBuffer, size, std::forward<TArgs>(params)...) };
        return this->eval(op);
    }

    /**
     * Eval sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job with a barrier.
     *
     * @param algorithm Algorithm to use for the record often used for OpAlgo
     * operations
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> eval(std::shared_ptr<Algorithm> algorithm,
                                   TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(algorithm,
                                     std::forward<TArgs>(params)...) };
        return this->eval(op);
    }

    /**
     * Eval Async sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job without a barrier. EvalAwait()
     * must ALWAYS be called after to ensure the sequence is terminated
     * correctly.
     *
     * @return Boolean stating whether execution was successful.
     */
    std::shared_ptr<Sequence> evalAsync();
    /**
     * Clears currnet operations to record provided one in the vector of
     * operations into the gpu as a submit job without a barrier. EvalAwait()
     * must ALWAYS be called after to ensure the sequence is terminated
     * correctly.
     *
     * @return Boolean stating whether execution was successful.
     */
    std::shared_ptr<Sequence> evalAsync(std::shared_ptr<OpBase> op);
    /**
     * Eval sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job with a barrier.
     *
     * @param tensors Vector of tensors to use for the operation
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> evalAsync(
      std::vector<std::shared_ptr<Tensor>> tensors,
      TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(tensors, std::forward<TArgs>(params)...) };
        return this->evalAsync(op);
    }
    /**
     * Eval sends all the recorded and stored operations in the vector of
     * operations into the gpu as a submit job with a barrier.
     *
     * @param algorithm Algorithm to use for the record often used for OpAlgo
     * operations
     * @param TArgs Template parameters that are used to initialise operation
     * which allows for extensible configurations on initialisation.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    template<typename T, typename... TArgs>
    std::shared_ptr<Sequence> evalAsync(std::shared_ptr<Algorithm> algorithm,
                                        TArgs&&... params)
    {
        std::shared_ptr<T> op{ new T(algorithm,
                                     std::forward<TArgs>(params)...) };
        return this->evalAsync(op);
    }

    /**
     * Eval Await waits for the fence to finish processing and then once it
     * finishes, it runs the postEval of all operations.
     *
     * @param waitFor Number of milliseconds to wait before timing out.
     * @return shared_ptr<Sequence> of the Sequence class itself
     */
    std::shared_ptr<Sequence> evalAwait(uint64_t waitFor = UINT64_MAX);

    /**
     * Clear function clears all operations currently recorded and starts
     * recording again.
     */
    void clear();

    /**
     * Return the timestamps that were latched at the beginning and
     * after each operation during the last eval() call.
     */
    std::vector<std::uint64_t> getTimestamps();

    /**
     * Begins recording commands for commands to be submitted into the command
     * buffer.
     */
    void begin();

    /**
     * Ends the recording and stops recording commands when the record command
     * is sent.
     */
    void end();

    /**
     * Returns true if the sequence is currently in recording activated.
     *
     * @return Boolean stating if recording ongoing.
     */
    bool isRecording() const;

    /**
     * Returns true if the sequence has been initialised, and it's based on the
     * GPU resources being referenced.
     *
     * @return Boolean stating if is initialized
     */
    bool isInit() const;

    /**
     * Clears command buffer and triggers re-record of all the current
     * operations saved, which is useful if the underlying kp::Tensors or
     * kp::Algorithms are modified and need to be re-recorded.
     */
    void rerecord();

    /**
     * Returns true if the sequence is currently running - mostly used for async
     * workloads.
     *
     * @return Boolean stating if currently running.
     */
    bool isRunning() const;

    /**
     * Destroys and frees the GPU resources which include the buffer and memory
     * and sets the sequence as init=False.
     */
    void destroy();

  private:
    // -------------- NEVER OWNED RESOURCES
    std::shared_ptr<vk::PhysicalDevice> mPhysicalDevice = nullptr;
    std::shared_ptr<vk::Device> mDevice = nullptr;
    std::shared_ptr<vk::Queue> mComputeQueue = nullptr;
    uint32_t mQueueIndex = -1;

    // -------------- OPTIONALLY OWNED RESOURCES
    std::shared_ptr<vk::CommandPool> mCommandPool = nullptr;
    bool mFreeCommandPool = false;
    std::shared_ptr<vk::CommandBuffer> mCommandBuffer = nullptr;
    bool mFreeCommandBuffer = false;

    // -------------- ALWAYS OWNED RESOURCES
    vk::Fence mFence;
    std::vector<std::shared_ptr<OpBase>> mOperations{};
    std::shared_ptr<vk::QueryPool> timestampQueryPool = nullptr;

    // State
    bool mRecording = false;
    bool mIsRunning = false;

    // Create functions
    void createCommandPool();
    void createCommandBuffer();
    void createTimestampQueryPool(uint32_t totalTimestamps);
};

} // End namespace kp
