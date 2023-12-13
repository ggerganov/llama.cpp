// SPDX-License-Identifier: Apache-2.0

#include "kompute/Tensor.hpp"

namespace kp {

std::string
Tensor::toString(Tensor::TensorDataTypes dt)
{
    switch (dt) {
        case TensorDataTypes::eBool:
            return "eBool";
        case TensorDataTypes::eInt:
            return "eInt";
        case TensorDataTypes::eUnsignedInt:
            return "eUnsignedInt";
        case TensorDataTypes::eFloat:
            return "eFloat";
        case TensorDataTypes::eDouble:
            return "eDouble";
        default:
            return "unknown";
    }
}

std::string
Tensor::toString(Tensor::TensorTypes dt)
{
    switch (dt) {
        case TensorTypes::eDevice:
            return "eDevice";
        case TensorTypes::eHost:
            return "eHost";
        case TensorTypes::eStorage:
            return "eStorage";
        default:
            return "unknown";
    }
}

Tensor::Tensor(std::shared_ptr<vk::PhysicalDevice> physicalDevice,
               std::shared_ptr<vk::Device> device,
               void* data,
               uint32_t elementTotalCount,
               uint32_t elementMemorySize,
               const TensorDataTypes& dataType,
               vk::DeviceMemory *primaryMemory,
               vk::Buffer *primaryBuffer,
               vk::DeviceMemory *stagingMemory,
               vk::Buffer *stagingBuffer,
               vk::DeviceSize offset,
               const TensorTypes& tensorType)
{
    KP_LOG_DEBUG("Kompute Tensor constructor data length: {}, and type: {}",
                 elementTotalCount,
                 Tensor::toString(tensorType));

    this->mPhysicalDevice = physicalDevice;
    this->mDevice = device;
    this->mDataType = dataType;
    this->mTensorType = tensorType;

    this->rebuild(data, elementTotalCount, elementMemorySize, primaryMemory, primaryBuffer, stagingMemory, stagingBuffer, offset);
}

Tensor::~Tensor()
{
    KP_LOG_DEBUG("Kompute Tensor destructor started. Type: {}",
                 Tensor::toString(this->tensorType()));

    if (this->mDevice) {
        this->destroy();
    }

    KP_LOG_DEBUG("Kompute Tensor destructor success");
}

void
Tensor::rebuild(void* /*data*/,
                uint32_t elementTotalCount,
                uint64_t memorySize,
                vk::DeviceMemory *primaryMemory,
                vk::Buffer *primaryBuffer,
                vk::DeviceMemory *stagingMemory,
                vk::Buffer *stagingBuffer,
                vk::DeviceSize offset)
{
    KP_LOG_DEBUG("Kompute Tensor rebuilding with size {}", elementTotalCount);

    this->mSize = elementTotalCount;
    this->mMemorySize = memorySize;
    this->mOffset = offset;

    if (this->mPrimaryBuffer || this->mPrimaryMemory) {
        KP_LOG_DEBUG(
          "Kompute Tensor destroying existing resources before rebuild");
        this->destroy();
    }

    this->setGPUResources(primaryMemory, primaryBuffer, stagingMemory, stagingBuffer, offset);
}

Tensor::TensorTypes
Tensor::tensorType()
{
    return this->mTensorType;
}

bool
Tensor::isInit()
{
    return this->mDevice && this->mPrimaryBuffer && this->mPrimaryMemory &&
           this->mRawData;
}

uint32_t
Tensor::size()
{
    return this->mSize;
}

uint64_t
Tensor::memorySize()
{
    return this->mMemorySize;
}

kp::Tensor::TensorDataTypes
Tensor::dataType()
{
    return this->mDataType;
}

void*
Tensor::rawData()
{
    return this->mRawData;
}

void
Tensor::setRawData(const void* data)
{
    memcpy(this->mRawData, data, this->memorySize());
}

void
Tensor::recordCopyFrom(const vk::CommandBuffer& commandBuffer,
                       std::shared_ptr<Tensor> copyFromTensor)
{

    vk::DeviceSize bufferSize(this->memorySize());
    vk::BufferCopy copyRegion(mOffset, mOffset, bufferSize);

    KP_LOG_DEBUG("Kompute Tensor recordCopyFrom data size {}.", bufferSize);

    this->recordCopyBuffer(commandBuffer,
                           copyFromTensor->mPrimaryBuffer,
                           this->mPrimaryBuffer,
                           bufferSize,
                           copyRegion);
}

void
Tensor::recordCopyFromStagingToDevice(const vk::CommandBuffer& commandBuffer)
{
    if (!this->mStagingBuffer)
        return;

    vk::DeviceSize bufferSize(this->memorySize());
    vk::BufferCopy copyRegion(mOffset, mOffset, bufferSize);

    KP_LOG_DEBUG("Kompute Tensor copying data size {}.", bufferSize);

    this->recordCopyBuffer(commandBuffer,
                           this->mStagingBuffer,
                           this->mPrimaryBuffer,
                           bufferSize,
                           copyRegion);
}

void
Tensor::recordCopyFromDeviceToStaging(const vk::CommandBuffer& commandBuffer)
{
    if (!this->mStagingBuffer)
        return;

    vk::DeviceSize bufferSize(this->memorySize());
    vk::BufferCopy copyRegion(mOffset, mOffset, bufferSize);

    KP_LOG_DEBUG("Kompute Tensor copying data size {}.", bufferSize);

    this->recordCopyBuffer(commandBuffer,
                           this->mPrimaryBuffer,
                           this->mStagingBuffer,
                           bufferSize,
                           copyRegion);
}

void
Tensor::recordCopyBuffer(const vk::CommandBuffer& commandBuffer,
                         vk::Buffer *bufferFrom,
                         vk::Buffer *bufferTo,
                         vk::DeviceSize /*bufferSize*/,
                         vk::BufferCopy copyRegion)
{

    commandBuffer.copyBuffer(*bufferFrom, *bufferTo, copyRegion);
}

void
Tensor::recordFill(const vk::CommandBuffer &commandBuffer,
                   uint32_t fill)
{
    commandBuffer.fillBuffer(*this->mPrimaryBuffer, mOffset, this->memorySize(), fill);
}

void
Tensor::recordPrimaryBufferMemoryBarrier(const vk::CommandBuffer& commandBuffer,
                                         vk::AccessFlagBits srcAccessMask,
                                         vk::AccessFlagBits dstAccessMask,
                                         vk::PipelineStageFlagBits srcStageMask,
                                         vk::PipelineStageFlagBits dstStageMask)
{
    KP_LOG_DEBUG("Kompute Tensor recording PRIMARY buffer memory barrier");

    this->recordBufferMemoryBarrier(commandBuffer,
                                    *this->mPrimaryBuffer,
                                    srcAccessMask,
                                    dstAccessMask,
                                    srcStageMask,
                                    dstStageMask);
}

void
Tensor::recordStagingBufferMemoryBarrier(const vk::CommandBuffer& commandBuffer,
                                         vk::AccessFlagBits srcAccessMask,
                                         vk::AccessFlagBits dstAccessMask,
                                         vk::PipelineStageFlagBits srcStageMask,
                                         vk::PipelineStageFlagBits dstStageMask)
{
    if (!this->mStagingBuffer)
        return;

    KP_LOG_DEBUG("Kompute Tensor recording STAGING buffer memory barrier");

    this->recordBufferMemoryBarrier(commandBuffer,
                                    *this->mStagingBuffer,
                                    srcAccessMask,
                                    dstAccessMask,
                                    srcStageMask,
                                    dstStageMask);
}

void
Tensor::recordBufferMemoryBarrier(const vk::CommandBuffer& commandBuffer,
                                  const vk::Buffer& buffer,
                                  vk::AccessFlagBits srcAccessMask,
                                  vk::AccessFlagBits dstAccessMask,
                                  vk::PipelineStageFlagBits srcStageMask,
                                  vk::PipelineStageFlagBits dstStageMask)
{
    KP_LOG_DEBUG("Kompute Tensor recording buffer memory barrier");

    vk::DeviceSize bufferSize = this->memorySize();

    vk::BufferMemoryBarrier bufferMemoryBarrier;
    bufferMemoryBarrier.buffer = buffer;
    bufferMemoryBarrier.size = bufferSize;
    bufferMemoryBarrier.srcAccessMask = srcAccessMask;
    bufferMemoryBarrier.dstAccessMask = dstAccessMask;
    bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    commandBuffer.pipelineBarrier(srcStageMask,
                                  dstStageMask,
                                  vk::DependencyFlags(),
                                  nullptr,
                                  bufferMemoryBarrier,
                                  nullptr);
}

vk::DescriptorBufferInfo
Tensor::constructDescriptorBufferInfo()
{
    KP_LOG_DEBUG("Kompute Tensor construct descriptor buffer info size {}",
                 this->memorySize());
    vk::DeviceSize bufferSize = this->memorySize();
    return vk::DescriptorBufferInfo(*this->mPrimaryBuffer,
                                    mOffset, // offset
                                    bufferSize);
}

vk::BufferUsageFlags
Tensor::getPrimaryBufferUsageFlags()
{
    switch (this->mTensorType) {
        case TensorTypes::eDevice:
            return vk::BufferUsageFlagBits::eStorageBuffer |
                   vk::BufferUsageFlagBits::eTransferSrc |
                   vk::BufferUsageFlagBits::eTransferDst;
            break;
        case TensorTypes::eHost:
            return vk::BufferUsageFlagBits::eStorageBuffer |
                   vk::BufferUsageFlagBits::eTransferSrc |
                   vk::BufferUsageFlagBits::eTransferDst;
            break;
        case TensorTypes::eStorage:
            return vk::BufferUsageFlagBits::eStorageBuffer;
            break;
        default:
            throw std::runtime_error("Kompute Tensor invalid tensor type");
    }
}

vk::MemoryPropertyFlags
Tensor::getPrimaryMemoryPropertyFlags()
{
    switch (this->mTensorType) {
        case TensorTypes::eDevice:
            return vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case TensorTypes::eHost:
            return vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        case TensorTypes::eStorage:
            return vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        default:
            throw std::runtime_error("Kompute Tensor invalid tensor type");
    }
}

vk::BufferUsageFlags
Tensor::getStagingBufferUsageFlags()
{
    switch (this->mTensorType) {
        case TensorTypes::eDevice:
            return vk::BufferUsageFlagBits::eTransferSrc |
                   vk::BufferUsageFlagBits::eTransferDst;
            break;
        default:
            throw std::runtime_error("Kompute Tensor invalid tensor type");
    }
}

vk::MemoryPropertyFlags
Tensor::getStagingMemoryPropertyFlags()
{
    switch (this->mTensorType) {
        case TensorTypes::eDevice:
            return vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        default:
            throw std::runtime_error("Kompute Tensor invalid tensor type");
    }
}

void
Tensor::setGPUResources(vk::DeviceMemory *primaryMemory,
                        vk::Buffer *primaryBuffer,
                        vk::DeviceMemory *stagingMemory,
                        vk::Buffer *stagingBuffer,
                        vk::DeviceSize /*offset*/)
{
    KP_LOG_DEBUG("Kompute Tensor creating buffer");

    if (!this->mPhysicalDevice) {
        throw std::runtime_error("Kompute Tensor phyisical device is null");
    }
    if (!this->mDevice) {
        throw std::runtime_error("Kompute Tensor device is null");
    }

    KP_LOG_DEBUG("Kompute Tensor creating primary buffer and memory");

    this->mPrimaryBuffer = primaryBuffer;
    this->mPrimaryMemory = primaryMemory;

    if (this->mTensorType == TensorTypes::eDevice) {
        KP_LOG_DEBUG("Kompute Tensor creating staging buffer and memory");

        this->mStagingBuffer = stagingBuffer;
        this->mStagingMemory = stagingMemory;
    }

    KP_LOG_DEBUG("Kompute Tensor buffer & memory creation successful");
}

void
Tensor::destroy()
{
    KP_LOG_DEBUG("Kompute Tensor started destroy()");

    // Setting raw data to null regardless whether device is available to
    // invalidate Tensor
    this->mRawData = nullptr;
    this->mSize = 0;
    this->mMemorySize = 0;

    if (!this->mDevice) {
        KP_LOG_WARN(
          "Kompute Tensor destructor reached with null Device pointer");
        return;
    }

    if (this->mDevice) {
        this->mDevice = nullptr;
    }

    KP_LOG_DEBUG("Kompute Tensor successful destroy()");
}

template<>
Tensor::TensorDataTypes
TensorT<bool>::dataType()
{
    return Tensor::TensorDataTypes::eBool;
}

template<>
Tensor::TensorDataTypes
TensorT<int32_t>::dataType()
{
    return Tensor::TensorDataTypes::eInt;
}

template<>
Tensor::TensorDataTypes
TensorT<uint32_t>::dataType()
{
    return Tensor::TensorDataTypes::eUnsignedInt;
}

template<>
Tensor::TensorDataTypes
TensorT<float>::dataType()
{
    return Tensor::TensorDataTypes::eFloat;
}

template<>
Tensor::TensorDataTypes
TensorT<double>::dataType()
{
    return Tensor::TensorDataTypes::eDouble;
}

}
