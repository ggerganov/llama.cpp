// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kompute/Core.hpp"
#include "logger/Logger.hpp"
#include <memory>
#include <string>

namespace kp {

/**
 * Structured data used in GPU operations.
 *
 * Tensors are the base building block in Kompute to perform operations across
 * GPUs. Each tensor would have a respective Vulkan memory and buffer, which
 * would be used to store their respective data. The tensors can be used for GPU
 * data storage or transfer.
 */
class Tensor
{
  public:
    /**
     * Type for tensors created: Device allows memory to be transferred from
     * staging buffers. Staging are host memory visible. Storage are device
     * visible but are not set up to transfer or receive data (only for shader
     * storage).
     */
    enum class TensorTypes
    {
        eDevice = 0,  ///< Type is device memory, source and destination
        eHost = 1,    ///< Type is host memory, source and destination
        eStorage = 2, ///< Type is Device memory (only)
    };
    enum class TensorDataTypes
    {
        eBool = 0,
        eInt = 1,
        eUnsignedInt = 2,
        eFloat = 3,
        eDouble = 4,
    };

    static std::string toString(TensorDataTypes dt);
    static std::string toString(TensorTypes dt);

    /**
     *  Constructor with data provided which would be used to create the
     * respective vulkan buffer and memory.
     *
     *  @param physicalDevice The physical device to use to fetch properties
     *  @param device The device to use to create the buffer and memory from
     *  @param data Non-zero-sized vector of data that will be used by the
     * tensor
     *  @param tensorTypes Type for the tensor which is of type TensorTypes
     */
    Tensor(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
           std::shared_ptr<vk::Device> device,
           void* data,
           uint32_t elementTotalCount,
           uint32_t memorySize,
           const TensorDataTypes& dataType,
           vk::DeviceMemory *primaryMemory,
           vk::Buffer *primaryBuffer,
           vk::DeviceMemory *stagingMemory,
           vk::Buffer *stagingBuffer,
           vk::DeviceSize offset,
           const TensorTypes& tensorType = TensorTypes::eDevice);

    /**
     * Destructor which is in charge of freeing vulkan resources unless they
     * have been provided externally.
     */
    virtual ~Tensor();

    /**
     * Function to trigger reinitialisation of the tensor buffer and memory with
     * new data as well as new potential device type.
     *
     * @param data Vector of data to use to initialise vector from
     * @param tensorType The type to use for the tensor
     */
    void rebuild(void* data,
                 uint32_t elementTotalCount,
                 uint64_t memorySize,
                 vk::DeviceMemory *primaryMemory,
                 vk::Buffer *primaryBuffer,
                 vk::DeviceMemory *stagingMemory,
                 vk::Buffer *stagingBuffer,
                 vk::DeviceSize offset);

    /**
     * Destroys and frees the GPU resources which include the buffer and memory.
     */
    void destroy();

    /**
     * Check whether tensor is initialized based on the created gpu resources.
     *
     * @returns Boolean stating whether tensor is initialized
     */
    bool isInit();

    /**
     * Retrieve the tensor type of the Tensor
     *
     * @return Tensor type of tensor
     */
    TensorTypes tensorType();

    /**
     * Records a copy from the memory of the tensor provided to the current
     * thensor. This is intended to pass memory into a processing, to perform
     * a staging buffer transfer, or to gather output (between others).
     *
     * @param commandBuffer Vulkan Command Buffer to record the commands into
     * @param copyFromTensor Tensor to copy the data from
     */
    void recordCopyFrom(const vk::CommandBuffer& commandBuffer,
                        std::shared_ptr<Tensor> copyFromTensor);

    void recordFill(const vk::CommandBuffer &commandBuffer,
                    uint32_t fill);

    /**
     * Records a copy from the internal staging memory to the device memory
     * using an optional barrier to wait for the operation. This function would
     * only be relevant for kp::Tensors of type eDevice.
     *
     * @param commandBuffer Vulkan Command Buffer to record the commands into
     */
    void recordCopyFromStagingToDevice(const vk::CommandBuffer& commandBuffer);

    /**
     * Records a copy from the internal device memory to the staging memory
     * using an optional barrier to wait for the operation. This function would
     * only be relevant for kp::Tensors of type eDevice.
     *
     * @param commandBuffer Vulkan Command Buffer to record the commands into
     */
    void recordCopyFromDeviceToStaging(const vk::CommandBuffer& commandBuffer);

    /**
     * Records the buffer memory barrier into the primary buffer and command
     * buffer which ensures that relevant data transfers are carried out
     * correctly.
     *
     * @param commandBuffer Vulkan Command Buffer to record the commands into
     * @param srcAccessMask Access flags for source access mask
     * @param dstAccessMask Access flags for destination access mask
     * @param scrStageMask Pipeline stage flags for source stage mask
     * @param dstStageMask Pipeline stage flags for destination stage mask
     */
    void recordPrimaryBufferMemoryBarrier(
      const vk::CommandBuffer& commandBuffer,
      vk::AccessFlagBits srcAccessMask,
      vk::AccessFlagBits dstAccessMask,
      vk::PipelineStageFlagBits srcStageMask,
      vk::PipelineStageFlagBits dstStageMask);
    /**
     * Records the buffer memory barrier into the staging buffer and command
     * buffer which ensures that relevant data transfers are carried out
     * correctly.
     *
     * @param commandBuffer Vulkan Command Buffer to record the commands into
     * @param srcAccessMask Access flags for source access mask
     * @param dstAccessMask Access flags for destination access mask
     * @param scrStageMask Pipeline stage flags for source stage mask
     * @param dstStageMask Pipeline stage flags for destination stage mask
     */
    void recordStagingBufferMemoryBarrier(
      const vk::CommandBuffer& commandBuffer,
      vk::AccessFlagBits srcAccessMask,
      vk::AccessFlagBits dstAccessMask,
      vk::PipelineStageFlagBits srcStageMask,
      vk::PipelineStageFlagBits dstStageMask);

    /**
     * Constructs a vulkan descriptor buffer info which can be used to specify
     * and reference the underlying buffer component of the tensor without
     * exposing it.
     *
     * @return Descriptor buffer info with own buffer
     */
    vk::DescriptorBufferInfo constructDescriptorBufferInfo();

    /**
     * Returns the size/magnitude of the Tensor, which will be the total number
     * of elements across all dimensions
     *
     * @return Unsigned integer representing the total number of elements
     */
    uint32_t size();

    /**
     * Returns the total memory size of the data contained by the Tensor object
     *
     * @return Unsigned integer representing the memory of the tensor in bytes.
     */
    uint64_t memorySize();

    /**
     * Retrieve the data type of the tensor (host, device, storage)
     *
     * @return Data type of tensor of type kp::Tensor::TensorDataTypes
     */
    TensorDataTypes dataType();

    /**
     * Retrieve the raw data via the pointer to the memory that contains the raw
     * memory of this current tensor. This tensor gets changed to a nullptr when
     * the Tensor is removed.
     *
     * @return Pointer to raw memory containing raw bytes data of Tensor.
     */
    void* rawData();

    /**
     * Sets / resets the data of the tensor which is directly done on the GPU
     * host visible memory available by the tensor.
     */
    void setRawData(const void* data);

    /**
     * Template to return the pointer data converted by specific type, which
     * would be any of the supported types including float, double, int32,
     * uint32 and bool.
     *
     * @return Pointer to raw memory containing raw bytes data of Tensor.
     */
    template<typename T>
    T* data()
    {
        return (T*)this->mRawData;
    }

    /**
     * Template to get the data of the current tensor as a vector of specific
     * type, which would be any of the supported types including float, double,
     * int32, uint32 and bool.
     *
     * @return Vector of type provided by template.
     */
    template<typename T>
    std::vector<T> vector()
    {
        return { (T*)this->mRawData, ((T*)this->mRawData) + this->size() };
    }

  protected:
    // -------------- ALWAYS OWNED RESOURCES
    TensorTypes mTensorType;
    TensorDataTypes mDataType;
    uint32_t mSize = 0;
    uint64_t mMemorySize = 0;
    vk::DeviceSize mOffset = 0;
    void* mRawData = nullptr;

  private:
    // -------------- NEVER OWNED RESOURCES
    std::shared_ptr<vk::PhysicalDevice> mPhysicalDevice;
    std::shared_ptr<vk::Device> mDevice;
    vk::Buffer *mPrimaryBuffer = nullptr;
    vk::Buffer *mStagingBuffer = nullptr;
    vk::DeviceMemory *mPrimaryMemory = nullptr;
    vk::DeviceMemory *mStagingMemory = nullptr;

    void setGPUResources(vk::DeviceMemory *primaryMemory,
                         vk::Buffer *primaryBuffer,
                         vk::DeviceMemory *stagingMemory,
                         vk::Buffer *stagingBuffer,
                         vk::DeviceSize offset);
    void recordCopyBuffer(const vk::CommandBuffer& commandBuffer,
                          vk::Buffer *bufferFrom,
                          vk::Buffer *bufferTo,
                          vk::DeviceSize bufferSize,
                          vk::BufferCopy copyRegion);

    void recordBufferMemoryBarrier(const vk::CommandBuffer& commandBuffer,
                                   const vk::Buffer& buffer,
                                   vk::AccessFlagBits srcAccessMask,
                                   vk::AccessFlagBits dstAccessMask,
                                   vk::PipelineStageFlagBits srcStageMask,
                                   vk::PipelineStageFlagBits dstStageMask);

    // Private util functions
    vk::BufferUsageFlags getPrimaryBufferUsageFlags();
    vk::MemoryPropertyFlags getPrimaryMemoryPropertyFlags();
    vk::BufferUsageFlags getStagingBufferUsageFlags();
    vk::MemoryPropertyFlags getStagingMemoryPropertyFlags();
};

template<typename T>
class TensorT : public Tensor
{

  public:
    ~TensorT() { KP_LOG_DEBUG("Kompute TensorT destructor"); }

    TensorDataTypes dataType();
};

} // End namespace kp
