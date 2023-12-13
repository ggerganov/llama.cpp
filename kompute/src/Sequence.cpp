// SPDX-License-Identifier: Apache-2.0

#include "kompute/Sequence.hpp"

namespace kp {

Sequence::Sequence(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                   std::shared_ptr<vk::Device> device,
                   std::shared_ptr<vk::Queue> computeQueue,
                   uint32_t queueIndex,
                   uint32_t totalTimestamps)
{
    KP_LOG_DEBUG("Kompute Sequence Constructor with existing device & queue");

    this->mPhysicalDevice = physicalDevice;
    this->mDevice = device;
    this->mComputeQueue = computeQueue;
    this->mQueueIndex = queueIndex;

    this->createCommandPool();
    this->createCommandBuffer();
    if (totalTimestamps > 0)
        this->createTimestampQueryPool(totalTimestamps +
                                       1); //+1 for the first one
}

Sequence::~Sequence()
{
    KP_LOG_DEBUG("Kompute Sequence Destructor started");

    if (this->mDevice) {
        this->destroy();
    }
}

void
Sequence::begin()
{
    KP_LOG_DEBUG("Kompute sequence called BEGIN");

    if (this->isRecording()) {
        KP_LOG_DEBUG("Kompute Sequence begin called when already recording");
        return;
    }

    if (this->isRunning()) {
        throw std::runtime_error(
          "Kompute Sequence begin called when sequence still running");
    }

    KP_LOG_INFO("Kompute Sequence command now started recording");
    this->mCommandBuffer->begin(vk::CommandBufferBeginInfo());
    this->mRecording = true;

    // latch the first timestamp before any commands are submitted
    if (this->timestampQueryPool)
        this->mCommandBuffer->writeTimestamp(
          vk::PipelineStageFlagBits::eAllCommands,
          *this->timestampQueryPool,
          0);
}

void
Sequence::end()
{
    KP_LOG_DEBUG("Kompute Sequence calling END");

    if (this->isRunning()) {
        throw std::runtime_error(
          "Kompute Sequence begin called when sequence still running");
    }

    if (!this->isRecording()) {
        KP_LOG_WARN("Kompute Sequence end called when not recording");
        return;
    } else {
        KP_LOG_INFO("Kompute Sequence command recording END");
        this->mCommandBuffer->end();
        this->mRecording = false;
    }
}

void
Sequence::clear()
{
    KP_LOG_DEBUG("Kompute Sequence calling clear");
    if (this->isRecording()) {
        this->end();
    }
}

std::shared_ptr<Sequence>
Sequence::eval()
{
    KP_LOG_DEBUG("Kompute sequence EVAL BEGIN");

    return this->evalAsync()->evalAwait();
}

std::shared_ptr<Sequence>
Sequence::eval(std::shared_ptr<OpBase> op)
{
    this->clear();
    return this->record(op)->eval();
}

std::shared_ptr<Sequence>
Sequence::evalAsync()
{
    if (this->isRecording()) {
        this->end();
    }

    if (this->mIsRunning) {
        throw std::runtime_error(
          "Kompute Sequence evalAsync called when an eval async was "
          "called without successful wait");
    }

    this->mIsRunning = true;

    for (size_t i = 0; i < this->mOperations.size(); i++) {
        this->mOperations[i]->preEval(*this->mCommandBuffer);
    }

    vk::SubmitInfo submitInfo(
      0, nullptr, nullptr, 1, this->mCommandBuffer.get());

    this->mFence = this->mDevice->createFence(vk::FenceCreateInfo());

    KP_LOG_DEBUG(
      "Kompute sequence submitting command buffer into compute queue");

    this->mComputeQueue->submit(1, &submitInfo, this->mFence);

    return shared_from_this();
}

std::shared_ptr<Sequence>
Sequence::evalAsync(std::shared_ptr<OpBase> op)
{
    this->clear();
    this->record(op);
    this->evalAsync();
    return shared_from_this();
}

std::shared_ptr<Sequence>
Sequence::evalAwait(uint64_t waitFor)
{
    if (!this->mIsRunning) {
        KP_LOG_WARN("Kompute Sequence evalAwait called without existing eval");
        return shared_from_this();
    }

    vk::Result result =
      this->mDevice->waitForFences(1, &this->mFence, VK_TRUE, waitFor);
    this->mDevice->destroy(
      this->mFence, (vk::Optional<const vk::AllocationCallbacks>)nullptr);

    this->mIsRunning = false;

    if (result == vk::Result::eTimeout) {
        KP_LOG_WARN("Kompute Sequence evalAwait reached timeout of {}",
                    waitFor);
        return shared_from_this();
    }

    for (size_t i = 0; i < this->mOperations.size(); i++) {
        this->mOperations[i]->postEval(*this->mCommandBuffer);
    }

    return shared_from_this();
}

bool
Sequence::isRunning() const
{
    return this->mIsRunning;
}

bool
Sequence::isRecording() const
{
    return this->mRecording;
}

bool
Sequence::isInit() const
{
    return this->mDevice && this->mCommandPool && this->mCommandBuffer &&
           this->mComputeQueue;
}

void
Sequence::rerecord()
{
    this->end();
    std::vector<std::shared_ptr<OpBase>> ops = this->mOperations;
    this->mOperations.clear();
    for (const std::shared_ptr<kp::OpBase>& op : ops) {
        this->record(op);
    }
}

void
Sequence::destroy()
{
    KP_LOG_DEBUG("Kompute Sequence destroy called");

    if (!this->mDevice) {
        KP_LOG_WARN("Kompute Sequence destroy called "
                    "with null Device pointer");
        return;
    }

    if (this->mFreeCommandBuffer) {
        KP_LOG_INFO("Freeing CommandBuffer");
        if (!this->mCommandBuffer) {
            KP_LOG_WARN("Kompute Sequence destroy called with null "
                        "CommandPool pointer");
            return;
        }
        this->mDevice->freeCommandBuffers(
          *this->mCommandPool, 1, this->mCommandBuffer.get());

        this->mCommandBuffer = nullptr;
        this->mFreeCommandBuffer = false;

        KP_LOG_DEBUG("Kompute Sequence Freed CommandBuffer");
    }

    if (this->mFreeCommandPool) {
        KP_LOG_INFO("Destroying CommandPool");
        if (this->mCommandPool == nullptr) {
            KP_LOG_WARN("Kompute Sequence destroy called with null "
                        "CommandPool pointer");
            return;
        }
        this->mDevice->destroy(
          *this->mCommandPool,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);

        this->mCommandPool = nullptr;
        this->mFreeCommandPool = false;

        KP_LOG_DEBUG("Kompute Sequence Destroyed CommandPool");
    }

    if (this->mOperations.size()) {
        KP_LOG_INFO("Kompute Sequence clearing operations buffer");
        this->mOperations.clear();
    }

    if (this->timestampQueryPool) {
        KP_LOG_INFO("Destroying QueryPool");
        this->mDevice->destroy(
          *this->timestampQueryPool,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);

        this->timestampQueryPool = nullptr;
        KP_LOG_DEBUG("Kompute Sequence Destroyed QueryPool");
    }

    if (this->mDevice) {
        this->mDevice = nullptr;
    }
    if (this->mPhysicalDevice) {
        this->mPhysicalDevice = nullptr;
    }
    if (this->mComputeQueue) {
        this->mComputeQueue = nullptr;
    }
}

std::shared_ptr<Sequence>
Sequence::record(std::shared_ptr<OpBase> op)
{
    KP_LOG_DEBUG("Kompute Sequence record function started");

    this->begin();

    KP_LOG_DEBUG(
      "Kompute Sequence running record on OpBase derived class instance");

    op->record(*this->mCommandBuffer);

    this->mOperations.push_back(op);

    if (this->timestampQueryPool)
        this->mCommandBuffer->writeTimestamp(
          vk::PipelineStageFlagBits::eAllCommands,
          *this->timestampQueryPool,
          this->mOperations.size());

    return shared_from_this();
}

void
Sequence::createCommandPool()
{
    KP_LOG_DEBUG("Kompute Sequence creating command pool");

    if (!this->mDevice) {
        throw std::runtime_error("Kompute Sequence device is null");
    }

    this->mFreeCommandPool = true;

    vk::CommandPoolCreateInfo commandPoolInfo(vk::CommandPoolCreateFlags(),
                                              this->mQueueIndex);
    this->mCommandPool = std::make_shared<vk::CommandPool>();
    this->mDevice->createCommandPool(
      &commandPoolInfo, nullptr, this->mCommandPool.get());
    KP_LOG_DEBUG("Kompute Sequence Command Pool Created");
}

void
Sequence::createCommandBuffer()
{
    KP_LOG_DEBUG("Kompute Sequence creating command buffer");
    if (!this->mDevice) {
        throw std::runtime_error("Kompute Sequence device is null");
    }
    if (!this->mCommandPool) {
        throw std::runtime_error("Kompute Sequence command pool is null");
    }

    this->mFreeCommandBuffer = true;

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo(
      *this->mCommandPool, vk::CommandBufferLevel::ePrimary, 1);

    this->mCommandBuffer = std::make_shared<vk::CommandBuffer>();
    this->mDevice->allocateCommandBuffers(&commandBufferAllocateInfo,
                                          this->mCommandBuffer.get());
    KP_LOG_DEBUG("Kompute Sequence Command Buffer Created");
}

void
Sequence::createTimestampQueryPool(uint32_t totalTimestamps)
{
    KP_LOG_DEBUG("Kompute Sequence creating query pool");
    if (!this->isInit()) {
        throw std::runtime_error(
          "createTimestampQueryPool() called on uninitialized Sequence");
    }
    if (!this->mPhysicalDevice) {
        throw std::runtime_error("Kompute Sequence physical device is null");
    }

    vk::PhysicalDeviceProperties physicalDeviceProperties =
      this->mPhysicalDevice->getProperties();

    if (physicalDeviceProperties.limits.timestampComputeAndGraphics) {
        vk::QueryPoolCreateInfo queryPoolInfo;
        queryPoolInfo.setQueryCount(totalTimestamps);
        queryPoolInfo.setQueryType(vk::QueryType::eTimestamp);
        this->timestampQueryPool = std::make_shared<vk::QueryPool>(
          this->mDevice->createQueryPool(queryPoolInfo));

        KP_LOG_DEBUG("Query pool for timestamps created");
    } else {
        throw std::runtime_error("Device does not support timestamps");
    }
}

std::vector<std::uint64_t>
Sequence::getTimestamps()
{
    if (!this->timestampQueryPool)
        throw std::runtime_error("Timestamp latching not enabled");

    const auto n = this->mOperations.size() + 1;
    std::vector<std::uint64_t> timestamps(n, 0);
    this->mDevice->getQueryPoolResults(
      *this->timestampQueryPool,
      0,
      n,
      timestamps.size() * sizeof(std::uint64_t),
      timestamps.data(),
      sizeof(uint64_t),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

    return timestamps;
}

}
