// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Core.hpp"

#include "fmt/format.h"
#include "kompute/Tensor.hpp"
#include "logger/Logger.hpp"

namespace kp {

/**
    Abstraction for compute shaders that are run on top of tensors grouped via
   ParameterGroups (which group descriptorsets)
*/
class Algorithm
{
  public:
    /**
     *  Main constructor for algorithm with configuration parameters to create
     *  the underlying resources.
     *
     *  @param device The Vulkan device to use for creating resources
     *  @param tensors (optional) The tensors to use to create the descriptor
     * resources
     *  @param spirv (optional) The spirv code to use to create the algorithm
     *  @param workgroup (optional) The kp::Workgroup to use for the dispatch
     * which defaults to kp::Workgroup(tensor[0].size(), 1, 1) if not set.
     *  @param specializationConstants (optional) The templatable param is to be
     * used to initialize the specialization constants which cannot be changed
     * once set.
     *  @param pushConstants (optional) This templatable param is to be used
     * when initializing the pipeline, which set the size of the push constants
     * - these can be modified but all new values must have the same data type
     * and length as otherwise it will result in errors.
     */
    template<typename S = float, typename P = float>
    Algorithm(std::shared_ptr<vk::Device> device,
              vk::PipelineCache *pipelineCache,
              vk::DescriptorPool *pool,
              const std::vector<std::shared_ptr<Tensor>>& tensors = {},
              const std::vector<uint32_t>& spirv = {},
              const Workgroup& workgroup = {},
              const std::vector<S>& specializationConstants = {},
              const std::vector<P>& pushConstants = {})
    {
        KP_LOG_DEBUG("Kompute Algorithm Constructor with device");

        this->mDevice = device;
        this->mPipelineCache = pipelineCache;
        this->mDescriptorPool = pool;

        if (tensors.size() && spirv.size()) {
            KP_LOG_INFO(
              "Kompute Algorithm initialising with tensor size: {} and "
              "spirv size: {}",
              tensors.size(),
              spirv.size());
            this->rebuild(tensors,
                          spirv,
                          workgroup,
                          specializationConstants,
                          pushConstants);
        } else {
            KP_LOG_INFO(
              "Kompute Algorithm constructor with empty tensors and or "
              "spirv so not rebuilding vulkan components");
        }
    }

    /**
     *  Rebuild function to reconstruct algorithm with configuration parameters
     * to create the underlying resources.
     *
     *  @param tensors The tensors to use to create the descriptor resources
     *  @param spirv The spirv code to use to create the algorithm
     *  @param workgroup (optional) The kp::Workgroup to use for the dispatch
     * which defaults to kp::Workgroup(tensor[0].size(), 1, 1) if not set.
     *  @param specializationConstants (optional) The std::vector<float> to use
     * to initialize the specialization constants which cannot be changed once
     * set.
     *  @param pushConstants (optional) The std::vector<float> to use when
     * initializing the pipeline, which set the size of the push constants -
     * these can be modified but all new values must have the same vector size
     * as this initial value.
     */
    template<typename S = float, typename P = float>
    void rebuild(const std::vector<std::shared_ptr<Tensor>>& tensors,
                 const std::vector<uint32_t>& spirv,
                 const Workgroup& workgroup = {},
                 const std::vector<S>& specializationConstants = {},
                 const std::vector<P>& pushConstants = {})
    {
        KP_LOG_DEBUG("Kompute Algorithm rebuild started");

        this->mTensors = tensors;
        this->mSpirv = spirv;

        if (specializationConstants.size()) {
            if (this->mSpecializationConstantsData) {
                free(this->mSpecializationConstantsData);
            }
            uint32_t memorySize =
              sizeof(decltype(specializationConstants.back()));
            uint32_t size = specializationConstants.size();
            uint32_t totalSize = size * memorySize;
            this->mSpecializationConstantsData = malloc(totalSize);
            memcpy(this->mSpecializationConstantsData,
                   specializationConstants.data(),
                   totalSize);
            this->mSpecializationConstantsDataTypeMemorySize = memorySize;
            this->mSpecializationConstantsSize = size;
        }

        if (pushConstants.size()) {
            if (this->mPushConstantsData) {
                free(this->mPushConstantsData);
            }
            uint32_t memorySize = sizeof(decltype(pushConstants.back()));
            uint32_t size = pushConstants.size();
            uint32_t totalSize = size * memorySize;
            this->mPushConstantsData = malloc(totalSize);
            memcpy(this->mPushConstantsData, pushConstants.data(), totalSize);
            this->mPushConstantsDataTypeMemorySize = memorySize;
            this->mPushConstantsSize = size;
        }

        this->setWorkgroup(
          workgroup, this->mTensors.size() ? this->mTensors[0]->size() : 1);

        // Descriptor pool is created first so if available then destroy all
        // before rebuild
        if (this->isInit()) {
            this->destroy();
        }

        this->createParameters();
        this->createShaderModule();
        this->createPipeline();
    }

    /**
     * Destructor for Algorithm which is responsible for freeing and desroying
     * respective pipelines and owned parameter groups.
     */
    ~Algorithm();

    /**
     * Records the dispatch function with the provided template parameters or
     * alternatively using the size of the tensor by default.
     *
     * @param commandBuffer Command buffer to record the algorithm resources to
     */
    void recordDispatch(const vk::CommandBuffer& commandBuffer);

    /**
     * Records command that binds the "core" algorithm components which consist
     * of binding the pipeline and binding the descriptorsets.
     *
     * @param commandBuffer Command buffer to record the algorithm resources to
     */
    void recordBindCore(const vk::CommandBuffer& commandBuffer);

    /**
     * Records command that binds the push constants to the command buffer
     * provided
     * - it is required that the pushConstants provided are of the same size as
     * the ones provided during initialization.
     *
     * @param commandBuffer Command buffer to record the algorithm resources to
     */
    void recordBindPush(const vk::CommandBuffer& commandBuffer);

    /**
     * function that checks all the gpu resource components to verify if these
     * have been created and returns true if all are valid.
     *
     * @returns returns true if the algorithm is currently initialized.
     */
    bool isInit();

    /**
     * Sets the work group to use in the recordDispatch
     *
     * @param workgroup The kp::Workgroup value to use to update the algorithm.
     * It must have a value greater than 1 on the x value (index 1) otherwise it
     * will be initialized on the size of the first tensor (ie.
     * this->mTensor[0]->size())
     */
    void setWorkgroup(const Workgroup& workgroup, uint32_t minSize = 1);
    /**
     * Sets the push constants to the new value provided to use in the next
     * bindPush()
     *
     * @param pushConstants The templatable vector is to be used to set the push
     * constants to use in the next bindPush(...) calls. The constants provided
     * must be of the same size as the ones created during initialization.
     */
    template<typename T>
    void setPushConstants(const std::vector<T>& pushConstants)
    {
        uint32_t memorySize = sizeof(decltype(pushConstants.back()));
        uint32_t size = pushConstants.size();
        this->setPushConstants(pushConstants.data(), size, memorySize);
    }

    void updateDescriptors(vk::DescriptorPool *pool)
    {
        this->mDescriptorPool = pool;
        this->setWorkgroup(
          this->mWorkgroup, this->mTensors.size() ? this->mTensors[0]->size() : 1);

        this->updateParameters(); // TODO: See if we can reduce this
    }

    /**
     * Sets the push constants to the new value provided to use in the next
     * bindPush() with the raw memory block location and memory size to be used.
     *
     * @param data The raw data point to copy the data from, without modifying
     * the pointer.
     * @param size The number of data elements provided in the data
     * @param memorySize The memory size of each of the data elements in bytes.
     */
    void setPushConstants(const void* data, uint32_t size, uint32_t memorySize)
    {

        uint32_t totalSize = memorySize * size;
        uint32_t previousTotalSize =
          this->mPushConstantsDataTypeMemorySize * this->mPushConstantsSize;

        if (totalSize != previousTotalSize) {
            throw std::runtime_error(fmt::format(
              "Kompute Algorithm push "
              "constant total memory size provided is {} but expected {} bytes",
              totalSize,
              previousTotalSize));
        }
        if (this->mPushConstantsData) {
            free(this->mPushConstantsData);
        }

        this->mPushConstantsData = malloc(totalSize);
        memcpy(this->mPushConstantsData, data, totalSize);
        this->mPushConstantsDataTypeMemorySize = memorySize;
        this->mPushConstantsSize = size;
    }

    /**
     * Gets the current workgroup from the algorithm.
     *
     * @param The kp::Constant to use to set the push constants to use in the
     * next bindPush(...) calls. The constants provided must be of the same size
     * as the ones created during initialization.
     */
    const Workgroup& getWorkgroup();
    /**
     * Gets the specialization constants of the current algorithm.
     *
     * @returns The std::vector<float> currently set for specialization
     * constants
     */
    template<typename T>
    const std::vector<T> getSpecializationConstants()
    {
        return { (T*)this->mSpecializationConstantsData,
                 ((T*)this->mSpecializationConstantsData) +
                   this->mSpecializationConstantsSize };
    }
    /**
     * Gets the specialization constants of the current algorithm.
     *
     * @returns The std::vector<float> currently set for push constants
     */
    template<typename T>
    const std::vector<T> getPushConstants()
    {
        return { (T*)this->mPushConstantsData,
                 ((T*)this->mPushConstantsData) + this->mPushConstantsSize };
    }
    /**
     * Gets the current tensors that are used in the algorithm.
     *
     * @returns The list of tensors used in the algorithm.
     */
    const std::vector<std::shared_ptr<Tensor>>& getTensors();
    void setTensors(const std::vector<std::shared_ptr<Tensor>>& tensors);

    void destroy();

  private:
    // -------------- NEVER OWNED RESOURCES
    std::shared_ptr<vk::Device> mDevice;
    std::vector<std::shared_ptr<Tensor>> mTensors;

    // -------------- OPTIONALLY OWNED RESOURCES
    std::shared_ptr<vk::DescriptorSetLayout> mDescriptorSetLayout;
    bool mFreeDescriptorSetLayout = false;
    vk::DescriptorPool *mDescriptorPool = nullptr;
    std::shared_ptr<vk::DescriptorSet> mDescriptorSet;
    bool mFreeDescriptorSet = false;
    std::shared_ptr<vk::ShaderModule> mShaderModule;
    bool mFreeShaderModule = false;
    std::shared_ptr<vk::PipelineLayout> mPipelineLayout;
    bool mFreePipelineLayout = false;
    vk::PipelineCache *mPipelineCache = nullptr;
    std::shared_ptr<vk::Pipeline> mPipeline;
    bool mFreePipeline = false;

    // -------------- ALWAYS OWNED RESOURCES
    std::vector<uint32_t> mSpirv;
    void* mSpecializationConstantsData = nullptr;
    uint32_t mSpecializationConstantsDataTypeMemorySize = 0;
    uint32_t mSpecializationConstantsSize = 0;
    void* mPushConstantsData = nullptr;
    uint32_t mPushConstantsDataTypeMemorySize = 0;
    uint32_t mPushConstantsSize = 0;
    Workgroup mWorkgroup;

    // Create util functions
    void createShaderModule();
    void createPipeline();

    // Parameters
    void freeParameters();
    void createParameters();
    void updateParameters();
};

} // End namespace kp
