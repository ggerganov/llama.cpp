// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <set>
#include <unordered_map>

#include "kompute/Core.hpp"

#include "kompute/Sequence.hpp"
#include "logger/Logger.hpp"

#define KP_DEFAULT_SESSION "DEFAULT"

namespace kp {

/**
    Base orchestrator which creates and manages device and child components
*/
class Manager
{
  public:
    /**
        Base constructor.
    */
    Manager();

    /**
     * Manager destructor which would ensure all owned resources are destroyed
     * unless explicitly stated that resources should not be destroyed or freed.
     */
    ~Manager();

    bool hasInstance() const {
        return this->mInstance.get();
    }

    bool hasDevice() const {
        return this->mDevice.get();
    }

    bool hasVulkan() const {
        return this->mDynamicLoader.get();
    }

    /**
     * Initialize a device.
     *
     * @param physicalDeviceIndex The index of the physical device to use
     * @param familyQueueIndices (Optional) List of queue indices to add for
     * explicit allocation
     * @param desiredExtensions The desired extensions to load from
     * physicalDevice
     */
    void initializeDevice(uint32_t physicalDeviceIndex,
            const std::vector<uint32_t>& familyQueueIndices = {},
            const std::vector<std::string>& desiredExtensions = {});

    /**
     * Create a managed sequence that will be destroyed by this manager
     * if it hasn't been destroyed by its reference count going to zero.
     *
     * @param queueIndex The queue to use from the available queues
     * @param nrOfTimestamps The maximum number of timestamps to allocate.
     * If zero (default), disables latching of timestamps.
     * @returns Shared pointer with initialised sequence
     */
    std::shared_ptr<Sequence> sequence(uint32_t queueIndex = 0,
                                       uint32_t totalTimestamps = 0);

    /**
     * Create a managed tensor that will be destroyed by this manager
     * if it hasn't been destroyed by its reference count going to zero.
     *
     * @param data The data to initialize the tensor with
     * @param tensorType The type of tensor to initialize
     * @returns Shared pointer with initialised tensor
     */
    template<typename T>
    std::shared_ptr<TensorT<T>> tensorT(
      const std::vector<T>& data,
       vk::DeviceMemory *primaryMemory,
       vk::Buffer *primaryBuffer,
       vk::DeviceMemory *stagingMemory,
       vk::Buffer *stagingBuffer,
      Tensor::TensorTypes tensorType = Tensor::TensorTypes::eDevice)
    {
        KP_LOG_DEBUG("Kompute Manager tensor creation triggered");

        std::shared_ptr<TensorT<T>> tensor{ new kp::TensorT<T>(
          this->mPhysicalDevice, this->mDevice, data, primaryMemory, primaryBuffer, stagingMemory, stagingBuffer, tensorType) };

        if (this->mManageResources) {
            this->mManagedTensors.push_back(tensor);
        }

        return tensor;
    }

    std::shared_ptr<Tensor> tensor(
      void* data,
      uint32_t elementTotalCount,
      uint64_t memorySize,
      const Tensor::TensorDataTypes& dataType,
      vk::DeviceMemory *primaryMemory,
      vk::Buffer *primaryBuffer,
      vk::DeviceMemory *stagingMemory,
      vk::Buffer *stagingBuffer,
      vk::DeviceSize offset,
      Tensor::TensorTypes tensorType = Tensor::TensorTypes::eDevice)
    {
        std::shared_ptr<Tensor> tensor{ new kp::Tensor(this->mPhysicalDevice,
                                                       this->mDevice,
                                                       data,
                                                       elementTotalCount,
                                                       memorySize,
                                                       dataType,
                                                       primaryMemory,
                                                       primaryBuffer,
                                                       stagingMemory,
                                                       stagingBuffer,
                                                       offset,
                                                       tensorType) };

        if (this->mManageResources) {
            this->mManagedTensors.push_back(tensor);
        }

        return tensor;
    }

    /**
     * Default non-template function that can be used to create algorithm
     * objects which provides default types to the push and spec constants as
     * floats.
     *
     * @param tensors (optional) The tensors to initialise the algorithm with
     * @param spirv (optional) The SPIRV bytes for the algorithm to dispatch
     * @param workgroup (optional) kp::Workgroup for algorithm to use, and
     * defaults to (tensor[0].size(), 1, 1)
     * @param specializationConstants (optional) float vector to use for
     * specialization constants, and defaults to an empty constant
     * @param pushConstants (optional) float vector to use for push constants,
     * and defaults to an empty constant
     * @returns Shared pointer with initialised algorithm
     */
    std::shared_ptr<Algorithm> algorithm(
      const std::string &name,
      vk::DescriptorPool *pool,
      const std::vector<std::shared_ptr<Tensor>>& tensors = {},
      const std::vector<uint32_t>& spirv = {},
      const Workgroup& workgroup = {},
      const std::vector<float>& specializationConstants = {},
      const std::vector<float>& pushConstants = {})
    {
        return this->algorithm<>(
          name, pool, tensors, spirv, workgroup, specializationConstants, pushConstants);
    }

    /**
     * Create a managed algorithm that will be destroyed by this manager
     * if it hasn't been destroyed by its reference count going to zero.
     *
     * @param tensors (optional) The tensors to initialise the algorithm with
     * @param spirv (optional) The SPIRV bytes for the algorithm to dispatch
     * @param workgroup (optional) kp::Workgroup for algorithm to use, and
     * defaults to (tensor[0].size(), 1, 1)
     * @param specializationConstants (optional) templatable vector parameter to
     * use for specialization constants, and defaults to an empty constant
     * @param pushConstants (optional) templatable vector parameter to use for
     * push constants, and defaults to an empty constant
     * @returns Shared pointer with initialised algorithm
     */
    template<typename S = float, typename P = float>
    std::shared_ptr<Algorithm> algorithm(
      const std::string &name,
      vk::DescriptorPool *pool,
      const std::vector<std::shared_ptr<Tensor>>& tensors,
      const std::vector<uint32_t>& spirv,
      const Workgroup& workgroup,
      const std::vector<S>& specializationConstants,
      const std::vector<P>& pushConstants)
    {

        KP_LOG_DEBUG("Kompute Manager algorithm creation triggered");

        std::shared_ptr<Algorithm> algorithm{ new kp::Algorithm(
          this->mDevice,
          mPipelineCache.get(),
          pool,
          tensors,
          spirv,
          workgroup,
          specializationConstants,
          pushConstants) };

        if (this->mManageResources) {
            this->mManagedAlgorithmsMap.insert({name, algorithm});
        }

        return algorithm;
    }

    bool hasAlgorithm(const std::string &name) const {
        return mManagedAlgorithmsMap.find(name) != mManagedAlgorithmsMap.end();
    }

    std::shared_ptr<Algorithm> getAlgorithm(const std::string &name) const {
        auto it = mManagedAlgorithmsMap.find(name);
        if (it != mManagedAlgorithmsMap.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * Destroy the GPU resources and all managed resources by manager.
     **/
    void destroy();
    /**
     * Run a pseudo-garbage collection to release all the managed resources
     * that have been already freed due to these reaching to zero ref count.
     **/
    void clear();

    /**
     * Information about the current device.
     *
     * @return vk::PhysicalDeviceProperties containing information about the
     *device
     **/
    vk::PhysicalDeviceProperties getDeviceProperties() const;

    /**
     * List the devices available in the current vulkan instance.
     *
     * @return vector of physical devices containing their respective properties
     **/
    std::vector<vk::PhysicalDevice> listDevices() const;

    /**
     * The current Vulkan instance.
     *
     * @return a shared pointer to the current Vulkan instance held by this
     *object
     **/
    std::shared_ptr<vk::Instance> getVkInstance() const;

    std::shared_ptr<vk::Device> device() const { return mDevice; }
    std::shared_ptr<vk::PhysicalDevice> physicalDevice() const { return mPhysicalDevice; }
    std::shared_ptr<vk::PipelineCache> pipelineCache() const { return mPipelineCache; }

  private:
    // -------------- OPTIONALLY OWNED RESOURCES
    std::shared_ptr<vk::Instance> mInstance = nullptr;
    bool mFreeInstance = false;
    std::shared_ptr<vk::PhysicalDevice> mPhysicalDevice = nullptr;
    std::shared_ptr<vk::Device> mDevice = nullptr;
    std::shared_ptr<vk::DynamicLoader> mDynamicLoader = nullptr;
    bool mFreeDevice = false;

    // -------------- ALWAYS OWNED RESOURCES
    std::vector<std::weak_ptr<Tensor>> mManagedTensors;
    std::vector<std::weak_ptr<Sequence>> mManagedSequences;
    std::unordered_map<std::string, std::shared_ptr<Algorithm>> mManagedAlgorithmsMap;

    std::vector<uint32_t> mComputeQueueFamilyIndices;
    std::vector<std::shared_ptr<vk::Queue>> mComputeQueues;
    std::shared_ptr<vk::PipelineCache> mPipelineCache;

    bool mManageResources = false;

#ifndef KOMPUTE_DISABLE_VK_DEBUG_LAYERS
    vk::DebugReportCallbackEXT mDebugReportCallback;
    vk::DispatchLoaderDynamic mDebugDispatcher;
#endif

    // Create functions
    void createInstance();
    void createDevice(const std::vector<uint32_t>& familyQueueIndices = {},
                      uint32_t physicalDeviceIndex = 0,
                      const std::vector<std::string>& desiredExtensions = {});
};

} // End namespace kp
