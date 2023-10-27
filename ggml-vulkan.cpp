/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

#include "ggml-vulkan.h"
#include "ggml.h"

// These are generated at build time by cmake custom command
#include "shaderop_scale.h"
#include "shaderop_add.h"
#include "shaderop_addrow.h"
#include "shaderop_mul.h"
#include "shaderop_mulrow.h"
#include "shaderop_silu.h"
#include "shaderop_relu.h"
#include "shaderop_gelu.h"
#include "shaderop_softmax.h"
#include "shaderop_norm.h"
#include "shaderop_rmsnorm.h"
#include "shaderop_diagmask.h"
#include "shaderop_mul_mat_f16.h"
#include "shaderop_mul_mat_q8_0.h"
#include "shaderop_mul_mat_q4_0.h"
#include "shaderop_mul_mat_q4_1.h"
#include "shaderop_mul_mat_q6_k.h"
#include "shaderop_mul_mat_mat_f32.h"
#include "shaderop_getrows_f16.h"
#include "shaderop_getrows_q4_0.h"
#include "shaderop_getrows_q4_1.h"
#include "shaderop_getrows_q6_k.h"
#include "shaderop_rope.h"
#include "shaderop_cpy_f16_f16.h"
#include "shaderop_cpy_f16_f32.h"
#include "shaderop_cpy_f32_f16.h"
#include "shaderop_cpy_f32_f32.h"

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <fstream>
#include <exception>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <immintrin.h>
#include <kompute/Kompute.hpp>

#define QK4_0 32
#define QR4_0 2
#define QK4_1 32
#define QK_NL 16

typedef ggml_fp16_t half;
struct ggml_kompute_context {
    bool hasH2DAll = false;
    std::vector<ggml_vk_memory> buffers;
    std::shared_ptr<vk::DescriptorPool> pool;
};

// FIXME: It would be good to consolidate the kompute manager and the kompute context into one object
// and consolidate the init functions and simplify object lifetime management. As it currently stands,
// we *have* to have the kompute manager no matter what for device discovery, but the kompute context
// is only created when a device is set and vulkan is explicitly turned on.
ggml_kompute_context *s_kompute_context = nullptr;
kp::Manager *komputeManager() {
    static kp::Manager *s_mgr = nullptr;
    if (s_mgr && !s_mgr->hasInstance()) {
        delete s_mgr;
        s_mgr = nullptr;
    }
    if (!s_mgr)
        s_mgr = new kp::Manager;
    return s_mgr;
}

#ifdef __linux__
__attribute__((constructor))
static void enable_sam() {
    setenv("RADV_PERFTEST", "sam", false);
}
#endif

static bool ggml_vk_checkPhysicalDeviceFeatures(vk::PhysicalDevice physicalDevice) {
    vk::PhysicalDeviceFeatures availableFeatures;
    physicalDevice.getFeatures(&availableFeatures);

    if (!availableFeatures.shaderInt16)
        return false;

    vk::PhysicalDeviceVulkan11Features availableFeatures11;
    vk::PhysicalDeviceVulkan12Features availableFeatures12;

    availableFeatures11.pNext = &availableFeatures12;
    availableFeatures12.pNext = nullptr;

    vk::PhysicalDeviceFeatures2 features2;
    features2.pNext = &availableFeatures11;

    physicalDevice.getFeatures2(&features2);

    if (!availableFeatures11.uniformAndStorageBuffer16BitAccess ||
        !availableFeatures11.storageBuffer16BitAccess) {
        return false;
    }

    if (!availableFeatures12.storageBuffer8BitAccess ||
        !availableFeatures12.uniformAndStorageBuffer8BitAccess ||
        !availableFeatures12.shaderFloat16 ||
        !availableFeatures12.shaderInt8) {
        return false;
    }

    return true;
}

static std::string ggml_vk_getVendorName(uint32_t vendorID) {
    switch (vendorID) {
        case 0x10DE:
            return "nvidia";
        case 0x1002:
            return "amd";
        case 0x8086:
            return "intel";
        default:
            return "unknown";
    }
}

std::vector<ggml_vk_device> ggml_vk_available_devices(size_t memoryRequired) {
    std::vector<ggml_vk_device> results;
    if (!komputeManager()->hasVulkan() || !komputeManager()->hasInstance())
        return results;

    std::vector<vk::PhysicalDevice> physicalDevices = komputeManager()->listDevices();
    uint32_t deviceCount = physicalDevices.size();

    if (deviceCount == 0)
        return results;

    std::unordered_map<std::string, size_t> count_by_name;

    for (uint32_t i = 0; i < deviceCount; i++) {
        VkPhysicalDeviceProperties properties = physicalDevices.at(i).getProperties();
        VkPhysicalDeviceMemoryProperties memoryProperties = physicalDevices.at(i).getMemoryProperties();
        const uint32_t major = VK_VERSION_MAJOR(properties.apiVersion);
        const uint32_t minor = VK_VERSION_MINOR(properties.apiVersion);
        if (major < 1 || minor < 2)
            continue;

        if (!ggml_vk_checkPhysicalDeviceFeatures(physicalDevices.at(i)))
            continue;

        size_t heapSize = 0;
        for (uint32_t j = 0; j < memoryProperties.memoryHeapCount; ++j) {
            VkMemoryHeap heap = memoryProperties.memoryHeaps[j];
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                heapSize = heap.size;
                break;
            }
        }

        if (heapSize < memoryRequired)
            continue;

        vk::PhysicalDeviceSubgroupProperties subgroupProperties;
        vk::PhysicalDeviceProperties2 deviceProperties2;
        deviceProperties2.pNext = &subgroupProperties;
        physicalDevices.at(i).getProperties2(&deviceProperties2);

        if (subgroupProperties.subgroupSize < 32)
            continue;

        ggml_vk_device d;
        d.index = i;
        d.type = properties.deviceType;
        d.heapSize = heapSize;
        d.name = properties.deviceName;
        d.subgroupSize = subgroupProperties.subgroupSize;
        size_t n_idx = ++count_by_name[d.name];
        if (n_idx > 1) {
            d.name += " (" + std::to_string(n_idx) + ")";
        }
        d.vendor = ggml_vk_getVendorName(properties.vendorID);
        results.push_back(d);
    }

    std::stable_sort(results.begin(), results.end(),
        [](const ggml_vk_device& lhs, const ggml_vk_device& rhs) -> bool {
            if (lhs.type != rhs.type) {
                if (lhs.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return true;
                if (rhs.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return false;

                if (lhs.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return true;
                if (rhs.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return false;
            }
            return lhs.heapSize < rhs.heapSize;
        }
    );

    return results;
}

static void ggml_vk_filterByVendor(std::vector<ggml_vk_device>& devices, const std::string& targetVendor) {
    devices.erase(
        std::remove_if(devices.begin(), devices.end(),
            [&targetVendor](const ggml_vk_device& device) {
                return device.vendor != targetVendor;
            }),
        devices.end()
    );
}

static void ggml_vk_filterByName(std::vector<ggml_vk_device>& devices, const std::string& targetName) {
    devices.erase(
        std::remove_if(devices.begin(), devices.end(),
            [&targetName](const ggml_vk_device& device) {
                return device.name != targetName;
            }),
        devices.end()
    );
}

bool ggml_vk_init_device(size_t memoryRequired, const std::string &device) {
    if (device.empty())
        return false;

    std::vector<ggml_vk_device> devices = ggml_vk_available_devices(memoryRequired);
    if (device == "gpu") {
        if (devices.size() != 0)
            return ggml_vk_init_device(devices.front());
    } else if (device == "amd" || device == "nvidia" || device == "intel") {
        ggml_vk_filterByVendor(devices, device);
        if (devices.size() != 0)
            return ggml_vk_init_device(devices.front());
    } else {
        ggml_vk_filterByName(devices, device);
        if (devices.size() != 0)
            return ggml_vk_init_device(devices.front());
    }

    return ggml_vk_has_device();
}

bool ggml_vk_init_device(const ggml_vk_device &device) {
    return ggml_vk_init_device(device.index);
}

bool ggml_vk_init_device(int device) {
    komputeManager()->initializeDevice(device, {},
                         {"VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage",
                          "VK_KHR_16bit_storage", "VK_KHR_shader_non_semantic_info"});
    return ggml_vk_has_device();
}

bool ggml_vk_free_device() {
    if (!ggml_vk_has_device())
        return false;
    komputeManager()->destroy();
    // FIXME: The lifetime of these two needs to be tied together as we're relying upon the fact
    // the llama_free(ctx) destroys this memory and we just set the singleton to nullptr here which
    // is very brittle
    s_kompute_context = nullptr;
    return true;
}

bool ggml_vk_has_vulkan() {
    return komputeManager()->hasVulkan();
}

bool ggml_vk_has_device() {
    return komputeManager()->hasDevice();
}

bool ggml_vk_using_vulkan() {
    return s_kompute_context != nullptr;
}

ggml_vk_device ggml_vk_current_device() {
    if (!komputeManager()->hasDevice())
        return ggml_vk_device();

    std::vector<ggml_vk_device> devices = ggml_vk_available_devices(0);
    ggml_vk_filterByName(devices, komputeManager()->physicalDevice()->getProperties().deviceName);
    return devices.front();
}

ggml_kompute_context *ggml_vk_init() {
    s_kompute_context = new ggml_kompute_context;
    return s_kompute_context;
}

bool ggml_vk_has_h2d_all(struct ggml_kompute_context * ctx) {
    return ctx->hasH2DAll;
}

void ggml_vk_free(struct ggml_kompute_context * ctx) {
    assert(ctx == s_kompute_context);
    s_kompute_context = nullptr;
    if (ctx != nullptr) {
        delete ctx;
    }
}

static
void ggml_vk_allocate_descriptor_pool(struct ggml_kompute_context * ctx, size_t size) {
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes = {
        vk::DescriptorPoolSize(
          vk::DescriptorType::eStorageBuffer,
          3 * size // Descriptor count is number of possible tensors to pass into an algorithm
          )
    };

    vk::DescriptorPoolCreateInfo descriptorPoolInfo(
      vk::DescriptorPoolCreateFlags(),
      size, // Max sets
      static_cast<uint32_t>(descriptorPoolSizes.size()),
      descriptorPoolSizes.data());

    ctx->pool = std::make_shared<vk::DescriptorPool>();
    vk::Result r = komputeManager()->device()->createDescriptorPool(
      &descriptorPoolInfo, nullptr, ctx->pool.get());
    if (r != vk::Result::eSuccess)
        std::cerr << "Error allocating descriptor pool" << vk::to_string(r);
}

static
void ggml_vk_free_descriptor_pool(struct ggml_kompute_context * ctx) {
    if (ctx->pool) {
        komputeManager()->device()->destroy(
          *ctx->pool,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
        ctx->pool = nullptr;
    }
}

static
vk::Buffer *ggml_vk_allocate_buffer(size_t size) {
    vk::BufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferSrc |
                             vk::BufferUsageFlagBits::eTransferDst;
    bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::Buffer *vkBuffer = new vk::Buffer;
    vk::Result r = komputeManager()->device()->createBuffer(&bufferCreateInfo, nullptr, vkBuffer);
    if (r != vk::Result::eSuccess)
        std::cerr << "Error allocating buffer " << vk::to_string(r) << std::endl;
    return vkBuffer;
}

static
vk::DeviceMemory *ggml_vk_allocate(size_t size, vk::MemoryPropertyFlags flags, vk::MemoryRequirements requirements, bool *isHostVisible) {

    uint32_t memoryTypeIndex = -1;
    bool memoryTypeIndexFound = false;
    vk::PhysicalDeviceMemoryProperties memoryProperties = komputeManager()->physicalDevice()->getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        const vk::MemoryType &memoryType = memoryProperties.memoryTypes[i];
        const vk::MemoryHeap &memoryHeap = memoryProperties.memoryHeaps[memoryType.heapIndex];
        if (memoryHeap.size < size) {
            continue;
        }

        if (requirements.memoryTypeBits & (1 << i)) {
            if (((memoryProperties.memoryTypes[i]).propertyFlags &
                 flags) == flags) {
                memoryTypeIndex = i;
                memoryTypeIndexFound = true;
                if (isHostVisible && (memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                    *isHostVisible = true;
                }
                break;
            }
        }
    }
    if (!memoryTypeIndexFound) {
        throw std::runtime_error(
          "Memory type index for buffer creation not found");
    }

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    vk::DeviceMemory *vkDeviceMemory =  new vk::DeviceMemory;
    vk::Result r = komputeManager()->device()->allocateMemory(&allocInfo, nullptr, vkDeviceMemory);
    if (r != vk::Result::eSuccess) {
        std::cerr << "Error allocating memory " << vk::to_string(r) << std::endl;
        throw std::runtime_error("Error allocating vulkan memory.");
    }
    return vkDeviceMemory;
}

size_t ggml_vk_aligned_offset(size_t offset) {

    static size_t minStorageBufferOffsetAlignment = 0;
    if (minStorageBufferOffsetAlignment == 0) {
        vk::PhysicalDeviceProperties deviceProperties;
        deviceProperties = komputeManager()->physicalDevice()->getProperties();
        vk::PhysicalDeviceLimits deviceLimits = deviceProperties.limits;
        minStorageBufferOffsetAlignment = deviceLimits.minStorageBufferOffsetAlignment;
    }

    // If offset is already aligned, return it directly
    if (offset % minStorageBufferOffsetAlignment == 0) {
        return offset;
    }

    // Otherwise, return the largest multiple of minStorageBufferOffsetAlignment less than offset
    return (offset / minStorageBufferOffsetAlignment) * minStorageBufferOffsetAlignment;
}

static void ggml_vk_h2d_buffer(const ggml_vk_memory &memory) {
    if (memory.stagingBuffer)
        komputeManager()->sequence()->eval<kp::OpBufferSyncDevice>(memory.primaryBuffer, memory.stagingBuffer, memory.size);
}

static void ggml_vk_d2h_buffer(const ggml_vk_memory &memory) {
    if (memory.stagingBuffer)
        komputeManager()->sequence()->eval<kp::OpBufferSyncLocal>(memory.primaryBuffer, memory.stagingBuffer, memory.size);
}

ggml_vk_memory ggml_vk_allocate(size_t size) {
    ggml_vk_memory memory;
    bool isHostVisible = false;
    {
        memory.primaryBuffer = ggml_vk_allocate_buffer(size);
        vk::MemoryRequirements memoryRequirements = komputeManager()->device()->getBufferMemoryRequirements(*memory.primaryBuffer);
        vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        memory.primaryMemory = ggml_vk_allocate(size, memoryPropertyFlags, memoryRequirements, &isHostVisible);
        komputeManager()->device()->bindBufferMemory(*memory.primaryBuffer, *memory.primaryMemory, 0);
        if (isHostVisible) {
            vk::Result r = komputeManager()->device()->mapMemory(*memory.primaryMemory, 0, size, vk::MemoryMapFlags(), &memory.data);
            if (r != vk::Result::eSuccess)
                std::cerr << "Error mapping memory" << vk::to_string(r);
        }
    }

    if (!isHostVisible) {
        memory.stagingBuffer = ggml_vk_allocate_buffer(size);
        vk::MemoryRequirements memoryRequirements = komputeManager()->device()->getBufferMemoryRequirements(*memory.stagingBuffer);
        vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                      vk::MemoryPropertyFlagBits::eHostCoherent |
                                                      vk::MemoryPropertyFlagBits::eHostCached;
        memory.stagingMemory = ggml_vk_allocate(size, memoryPropertyFlags, memoryRequirements, &isHostVisible);
        komputeManager()->device()->bindBufferMemory(*memory.stagingBuffer, *memory.stagingMemory, 0);
        vk::Result r = komputeManager()->device()->mapMemory(*memory.stagingMemory, 0, size, vk::MemoryMapFlags(), &memory.data);
        if (r != vk::Result::eSuccess)
            std::cerr << "Error mapping memory" << vk::to_string(r);
    }

    memory.size = size;
    return memory;
}

void ggml_vk_free_memory(ggml_vk_memory &memory)
{
    komputeManager()->device()->destroy(
      *memory.primaryBuffer,
      (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    if (memory.stagingBuffer) {
        komputeManager()->device()->destroy(
          *memory.stagingBuffer,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    }
    komputeManager()->device()->freeMemory(
      *memory.primaryMemory,
      (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    if (memory.stagingMemory) {
        komputeManager()->device()->freeMemory(
          *memory.stagingMemory,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    }
}

static
decltype(ggml_kompute_context::buffers)::iterator ggml_vk_find_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t, uint64_t & offset) {
    for (auto it = ctx->buffers.begin(); ; it++) {
        if (it == ctx->buffers.end()) {
            fprintf(stderr, "%s: Failed to find tensor %p\n", __func__, t->data);
            return it;
        }
        if (it->data <= t->data &&
                reinterpret_cast<intptr_t>(it->data) + it->size >= (reinterpret_cast<intptr_t>(t->data) + ggml_nbytes(t))) {
            offset = reinterpret_cast<intptr_t>(t->data) - reinterpret_cast<intptr_t>(it->data);
            return it;
        }
    }
}

static
const std::shared_ptr<kp::Tensor> ggml_vk_get_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t, uint32_t *alignedOffset) {
    uint64_t originalOffset = 0;
    auto res = ggml_vk_find_tensor(ctx, t, originalOffset);
    if (res == ctx->buffers.end()) {
        static std::shared_ptr<kp::Tensor> nullTensor = nullptr;
        return nullTensor;
    }

    // Create a tensor whose memory will be composed of our buffers at the correct offset
    const size_t nelements = ggml_nelements(t);
    size_t nbytes = ggml_nbytes(t);

    size_t vulkanOffset = ggml_vk_aligned_offset(originalOffset);
    if (alignedOffset) {
        *alignedOffset = originalOffset - vulkanOffset;
        nbytes += *alignedOffset;
    }

    return komputeManager()->tensor(
        t->data,
        nelements,
        nbytes, kp::Tensor::TensorDataTypes::eFloat,
        res->primaryMemory, res->primaryBuffer,
        res->stagingMemory, res->stagingBuffer,
        vulkanOffset);
}

void ggml_vk_add_buffer(
        struct ggml_kompute_context * ctx,
        const char * /*name*/,
        const ggml_vk_memory &memory) {
    ctx->buffers.emplace_back(memory);
}

void ggml_vk_h2d_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    const auto res = ggml_vk_get_tensor(ctx, t, nullptr);
    GGML_ASSERT(res);
    komputeManager()->sequence()->eval<kp::OpTensorSyncDevice>({res});
}

void ggml_vk_h2d_all(struct ggml_kompute_context * ctx) {
    for (auto& it : ctx->buffers) {
        ggml_vk_h2d_buffer(it);
    }
    ctx->hasH2DAll = true;
}

void ggml_vk_d2h_all(struct ggml_kompute_context * ctx) {
    for (auto& it : ctx->buffers) {
        ggml_vk_d2h_buffer(it);
    }
}

void ggml_vk_d2h_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t) {
    const auto res = ggml_vk_get_tensor(ctx, t, nullptr);

    GGML_ASSERT(res);
    komputeManager()->sequence()->eval<kp::OpTensorSyncLocal>({res});
}

std::vector<uint32_t> getSpirvShader(const unsigned char* rawData, size_t size) {
    if (size % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Invalid size: must be divisible by sizeof(uint32_t)");
    }

    const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(rawData);
    size_t count = size / sizeof(uint32_t);
    return std::vector<uint32_t>(data_ptr, data_ptr + count);
}

inline static
uint32_t safe_divide(uint32_t a, uint32_t b) {
    if (b <= 1) {
        return a;
    }
    if ((a % b) != 0) {
        fprintf(stderr, "((%u %% %u) == %u) != 0\n", a, b, a % b);
        GGML_ASSERT(!"safe_divide result would've had remainder");
    }
    return a / b;
}

void ggml_vk_add(kp::Sequence& seq,
                    const std::shared_ptr<kp::Tensor>& inA,
                    const std::shared_ptr<kp::Tensor>& inB,
                    const std::shared_ptr<kp::Tensor>& out,
                    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                    uint32_t size) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_add_comp_spv,
        kp::shader_data::op_add_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4)
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_addrow(kp::Sequence& seq,
                 const std::shared_ptr<kp::Tensor>& inA,
                 const std::shared_ptr<kp::Tensor>& inB,
                 const std::shared_ptr<kp::Tensor>& out,
                 uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                 uint32_t size, uint32_t row = 0) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_addrow_comp_spv,
        kp::shader_data::op_addrow_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        uint32_t row;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        row
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_mul(kp::Sequence& seq,
                    const std::shared_ptr<kp::Tensor>& inA,
                    const std::shared_ptr<kp::Tensor>& inB,
                    const std::shared_ptr<kp::Tensor>& out,
                    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                    uint32_t size) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_comp_spv,
        kp::shader_data::op_mul_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4)
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_mulrow(kp::Sequence& seq,
                 const std::shared_ptr<kp::Tensor>& inA,
                 const std::shared_ptr<kp::Tensor>& inB,
                 const std::shared_ptr<kp::Tensor>& out,
                 uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                 uint32_t size, uint32_t row = 0) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_mulrow_comp_spv,
        kp::shader_data::op_mulrow_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        uint32_t row;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        row
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_scale(kp::Sequence& seq,
                   const std::shared_ptr<kp::Tensor>& in,
                   const std::shared_ptr<kp::Tensor>& out,
                   uint32_t inOff, uint32_t outOff,
                   uint32_t size, float scale) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_scale_comp_spv,
        kp::shader_data::op_scale_comp_spv_len);

    struct PushConstants {
        uint32_t inOff, outOff;
        float scale;
    } const pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        scale
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_xxlu(const std::vector<uint32_t>& spirv, kp::Sequence& seq,
                  const std::shared_ptr<kp::Tensor>& in,
                  const std::shared_ptr<kp::Tensor>& out,
                  uint32_t inOff, uint32_t outOff,
                  uint32_t size) {
    struct PushConstants {
        uint32_t inOff, outOff;
    } const pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
void ggml_vk_silu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_silu_comp_spv,
        kp::shader_data::op_silu_comp_spv_len);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_relu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_relu_comp_spv,
        kp::shader_data::op_relu_comp_spv_len);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_gelu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_gelu_comp_spv,
        kp::shader_data::op_gelu_comp_spv_len);

    ggml_vk_xxlu(spirv, std::forward<Args>(args)...);
}

void ggml_vk_soft_max(kp::Sequence& seq,
                      const std::shared_ptr<kp::Tensor>& in,
                      const std::shared_ptr<kp::Tensor>& out,
                      uint32_t inOff, uint32_t outOff,
                      int32_t ne00, int32_t ne01, int32_t ne02, uint32_t ne03) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_softmax_comp_spv,
        kp::shader_data::op_softmax_comp_spv_len);

    struct PushConstants {
        uint32_t inOff, outOff;
        int32_t ne00, ne01, ne02;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        // FIXME: The softmax kernel needs to be fixed to use the subgroupsize which can vary by device
        const uint32_t local_x = 32;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_norm_(const std::vector<uint32_t>& spirv, kp::Sequence& seq,
                   const std::shared_ptr<kp::Tensor>& in,
                   const std::shared_ptr<kp::Tensor>& out,
                   uint32_t inOff, uint32_t outOff,
                   int32_t ne00, int32_t nb01,
                   int32_t nrows, float epsilon) {
    GGML_ASSERT(nb01%sizeof(float) == 0);
    GGML_ASSERT(ne00%sizeof(float) == 0);

    struct PushConstants {
        uint32_t inOff, outOff;
        uint32_t ne00, nb01;
        float eps;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        (uint32_t)ne00, (uint32_t)nb01, epsilon
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {(uint32_t)nrows}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({(uint32_t)nrows});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
void ggml_vk_norm(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_norm_comp_spv,
        kp::shader_data::op_norm_comp_spv_len);

    ggml_vk_norm_(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_rms_norm(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_rmsnorm_comp_spv,
        kp::shader_data::op_rmsnorm_comp_spv_len);

    ggml_vk_norm_(spirv, std::forward<Args>(args)...);
}

void ggml_vk_diag_mask_inf(kp::Sequence& seq,
                           const std::shared_ptr<kp::Tensor>& in,
                           const std::shared_ptr<kp::Tensor>& out,
                           uint32_t inOff, uint32_t outOff,
                           uint32_t n_past,
                           int32_t ne00, int32_t ne01, int32_t ne02) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_diagmask_comp_spv,
        kp::shader_data::op_diagmask_comp_spv_len);

    struct PushConstants {
        uint32_t inOff, outOff;
        uint32_t n_past;
        int32_t ne00, ne01;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        n_past,
        ne00, ne01
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne00), unsigned(ne01), unsigned(ne02)}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne00), unsigned(ne01), unsigned(ne02)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_mul_mat_f16(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA,
                         const std::shared_ptr<kp::Tensor>& inB,
                         const std::shared_ptr<kp::Tensor>& out,
                         uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                         int32_t ne00, int32_t ne01, int32_t ne02,
                         uint32_t nb01, uint32_t nb02,
                         int32_t ne11, int32_t ne12,
                         uint32_t nb11, uint32_t nb12,
                         int32_t ne0, int32_t ne1) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_f16_comp_spv,
        kp::shader_data::op_mul_mat_f16_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00;
        uint32_t nb01, nb02;
        uint32_t nb11, nb12;
        int32_t ne02, ne12;
        int32_t ne0, ne1;
    } pushConsts {
        safe_divide(inAOff, 2), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, nb01, nb02, nb11, nb12, ne02, ne12, ne0, ne1,
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned(ne01), unsigned(ne11), unsigned(std::max(ne12, ne02))}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne11), unsigned(std::max(ne12, ne02))});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_mul_mat_q8_0(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA,
                         const std::shared_ptr<kp::Tensor>& inB,
                         const std::shared_ptr<kp::Tensor>& out,
                         uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                         int32_t ne00, int32_t ne01,
                         uint32_t nb01, uint32_t nb02,
                         int32_t ne11, int32_t ne12,
                         uint32_t nb11, uint32_t nb12,
                         int32_t ne0, int32_t ne1) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q8_0_comp_spv,
        kp::shader_data::op_mul_mat_q8_0_comp_spv_len);
    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00;
        uint32_t nb01, nb02;
        uint32_t nb11, nb12;
        int32_t ne0, ne1;
    } pushConsts {
        inAOff, safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, nb01, nb02, nb11, nb12, ne0, ne1,
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned(ne01), unsigned(ne11), unsigned(ne12)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne11), unsigned(ne12)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}


void ggml_vk_mul_mat_mat_f32(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA,
                         const std::shared_ptr<kp::Tensor>& inB,
                         const std::shared_ptr<kp::Tensor>& out,
                         uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                         int32_t ne00, int32_t ne01, int32_t ne02,
                         uint32_t nb01, uint32_t nb02,
                         int32_t ne11, int32_t ne12,
                         uint32_t nb11, uint32_t nb12,
                         uint32_t nb1, uint32_t nb2) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_mat_f32_comp_spv,
        kp::shader_data::op_mul_mat_mat_f32_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne01, ne02, ne11, ne12;
        uint32_t nb01, nb02;
        uint32_t nb11, nb12;
        uint32_t nb1, nb2;
    } pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02, ne11, ne12,
        nb01, nb02, nb11, nb12,
        nb1, nb2
    };

    const uint32_t local_x = ggml_vk_current_device().subgroupSize;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(),
        {inA, inB, out}, spirv,
        {unsigned(ne01),
         unsigned(ne11),
         unsigned(std::max(ne12, ne02))
         },
        {local_x},
        {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01),
                              unsigned(ne11),
                              unsigned(std::max(ne12, ne02)),
                              });
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_mul_mat_q4_x(const std::vector<uint32_t>& spirv, uint32_t block_size, kp::Sequence& seq,
                          const std::shared_ptr<kp::Tensor>& inA,
                          const std::shared_ptr<kp::Tensor>& inB,
                          const std::shared_ptr<kp::Tensor>& out,
                          uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                          int32_t ne00, int32_t ne10, int32_t ne0, int32_t ne1,
                          int32_t ne01, int32_t ne11, int32_t ne12, int32_t ne02) {
    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne10, ne0, ne1, ne01, gqa;
    } pushConsts {
        safe_divide(inAOff, block_size), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne10, ne0, ne1, ne01, ne12/ne02
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned((ne01 + 7)/8), unsigned(ne11), unsigned(ne12)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned((ne01 + 7)/8), unsigned(ne11), unsigned(ne12)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
void ggml_vk_mul_mat_q4_0(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q4_0_comp_spv,
        kp::shader_data::op_mul_mat_q4_0_comp_spv_len);

    ggml_vk_mul_mat_q4_x(spirv, 1/*We access blocks unaligned*/, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_mul_mat_q4_1(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q4_1_comp_spv,
        kp::shader_data::op_mul_mat_q4_1_comp_spv_len);

    ggml_vk_mul_mat_q4_x(spirv, 1/*We access blocks unaligned*/, std::forward<Args>(args)...);
}

void ggml_vk_mul_mat_q6_k(kp::Sequence& seq,
                          const std::shared_ptr<kp::Tensor>& inA,
                          const std::shared_ptr<kp::Tensor>& inB,
                          const std::shared_ptr<kp::Tensor>& out,
                          uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                          int32_t ne00, int32_t ne10, int32_t ne0, int32_t ne1,
                          int32_t ne01, int32_t ne11, int32_t ne12, int32_t ne02) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q6_k_comp_spv,
        kp::shader_data::op_mul_mat_q6_k_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne10, ne0, ne1, ne01, gqa;
    } pushConsts {
        inAOff, safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne10, ne0, ne1, ne01, ne12/ne02
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned((ne01 + 1)/2), unsigned(ne11), unsigned(ne12)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned((ne01 + 1)/2), unsigned(ne11), unsigned(ne12)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

void ggml_vk_get_rows(const std::vector<uint32_t>& spirv,
                      unsigned element_size, unsigned qk,
                      kp::Sequence& seq,
                      const std::shared_ptr<kp::Tensor>& inA,
                      const std::shared_ptr<kp::Tensor>& inB,
                      const std::shared_ptr<kp::Tensor>& out,
                      uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                      int32_t ne00, int32_t nb01, int32_t nb1,
                      uint32_t size) {
    GGML_ASSERT(nb01%element_size == 0);
    GGML_ASSERT(nb1%sizeof(float) == 0);
    if (qk) GGML_ASSERT(ne00%qk == 0);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, nb01, nb1;
    } pushConsts {
        safe_divide(inAOff, element_size), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, nb01, nb1
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
void ggml_vk_get_rows_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_f16_comp_spv,
        kp::shader_data::op_getrows_f16_comp_spv_len);

    ggml_vk_get_rows(spirv, sizeof(half), 0, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_get_rows_q4_0(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q4_0_comp_spv,
        kp::shader_data::op_getrows_q4_0_comp_spv_len);

    ggml_vk_get_rows(spirv, 1/*We access blocks unaligned*/, QK4_0, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_get_rows_q4_1(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q4_1_comp_spv,
        kp::shader_data::op_getrows_q4_1_comp_spv_len);

    ggml_vk_get_rows(spirv, 1/*We access blocks unaligned*/, QK4_1, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_get_rows_q6_k(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q6_k_comp_spv,
        kp::shader_data::op_getrows_q6_k_comp_spv_len);
    ggml_vk_get_rows(spirv, 1/*We access blocks unaligned*/, QK_NL, std::forward<Args>(args)...);
}

void ggml_vk_rope(kp::Sequence& seq,
                  const std::shared_ptr<kp::Tensor>& in,
                  const std::shared_ptr<kp::Tensor>& out,
                  uint32_t inOff, uint32_t outOff,
                  uint32_t n_past, int32_t n_dims, int32_t mode,
                  float freq_base, float freq_scale,
                  int32_t ne01, int32_t ne02, int32_t ne03,
                  uint32_t nb00, uint32_t nb01, uint32_t nb02, uint32_t nb03,
                  int32_t ne0,
                  uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_rope_comp_spv,
        kp::shader_data::op_rope_comp_spv_len);

    GGML_ASSERT(nb03%sizeof(float) == 0);
    GGML_ASSERT(nb02%sizeof(float) == 0);
    GGML_ASSERT(nb01%sizeof(float) == 0);
    GGML_ASSERT(nb00%sizeof(float) == 0);
    GGML_ASSERT(nb3%sizeof(float) == 0);
    GGML_ASSERT(nb2%sizeof(float) == 0);
    GGML_ASSERT(nb1%sizeof(float) == 0);
    GGML_ASSERT(nb0%sizeof(float) == 0);

    struct PushConstants {
        uint32_t inOff, outOff;
        uint32_t n_past;
        int32_t n_dims, mode;
        float freq_base, freq_scale;
        uint32_t nb00, nb01, nb02, nb03;
        int32_t ne0;
        uint32_t nb0, nb1, nb2, nb3;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        n_past, n_dims, mode,
        freq_base, freq_scale,
        nb00, nb01, nb02, nb03,
        ne0,
        nb0, nb1, nb2, nb3
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template<uint32_t in_element_size, uint32_t out_element_size>
void ggml_vk_cpy(const std::vector<uint32_t>& spirv,
                 kp::Sequence& seq,
                 const std::shared_ptr<kp::Tensor>& in,
                 const std::shared_ptr<kp::Tensor>& out,
                 uint32_t inOff, uint32_t outOff,
                 int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
                 uint32_t nb00, uint32_t nb01, uint32_t nb02, uint32_t nb03,
                 int32_t ne0, int32_t ne1, int32_t ne2,
                 uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3) {
    struct PushConstants {
        uint32_t inOff, outOff;
        int32_t ne00, ne01, ne02;
        uint32_t nb00, nb01, nb02, nb03;
        int32_t ne0, ne1, ne2;
        uint32_t nb0, nb1, nb2, nb3;
    } pushConsts {
        safe_divide(inOff, in_element_size), safe_divide(outOff, out_element_size),
        ne00, ne01, ne02,
        nb00, nb01, nb02, nb03,
        ne0, ne1, ne2,
        nb0, nb1, nb2, nb3
    };

    static std::string unique_name = std::string(__func__) +
                                     "_i_" + std::to_string(in_element_size) +
                                     "_o_" + std::to_string(out_element_size);
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(unique_name))
        s_algo = komputeManager()->algorithm<float, PushConstants>(unique_name, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(unique_name);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
void ggml_vk_cpy_f32_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f32_f16_comp_spv,
        kp::shader_data::op_cpy_f32_f16_comp_spv_len);
    ggml_vk_cpy<4, 2>(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_cpy_f32_f32(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f32_f32_comp_spv,
        kp::shader_data::op_cpy_f32_f32_comp_spv_len);
    ggml_vk_cpy<4, 4>(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_cpy_f16_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f16_f16_comp_spv,
        kp::shader_data::op_cpy_f16_f16_comp_spv_len);
    ggml_vk_cpy<2, 2>(spirv, std::forward<Args>(args)...);
}

template <typename... Args>
void ggml_vk_cpy_f16_f32(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f16_f32_comp_spv,
        kp::shader_data::op_cpy_f16_f32_comp_spv_len);
    ggml_vk_cpy<2, 4>(spirv, std::forward<Args>(args)...);
}

void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf) {
    const int n_seq = 8;

    // FIXME: Figure out if we can somehow optimize the size of the pool... right now we're setting
    // it to the size of the graph, but I think it can be made smaller?
    ggml_vk_allocate_descriptor_pool(ctx, gf->n_nodes);

    std::vector<std::shared_ptr<kp::Sequence>> sequences(n_seq);

    for (auto& sequence : sequences) {
        sequence = komputeManager()->sequence();
    }
    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int n_nodes_per_seq = (gf->n_nodes + n_seq - 1) / n_seq;

        auto& seq = *sequences[seq_idx];

        const int node_start = (seq_idx + 0) * n_nodes_per_seq;
        const int node_end = (seq_idx == n_seq - 1) ? gf->n_nodes : (seq_idx + 1) * n_nodes_per_seq;

        for (int i = node_start; i < node_end; ++i) {
            struct ggml_tensor * src0 = gf->nodes[i]->src[0];
            struct ggml_tensor * src1 = gf->nodes[i]->src[1];
            struct ggml_tensor * dst = gf->nodes[i];
            GGML_ASSERT(dst->data != nullptr);

            const int32_t ne00 = src0 ? src0->ne[0] : 0;
            const int32_t ne01 = src0 ? src0->ne[1] : 0;
            const int32_t ne02 = src0 ? src0->ne[2] : 0;
            const int32_t ne03 = src0 ? src0->ne[3] : 0;

            const uint32_t nb00 = src0 ? src0->nb[0] : 0;
            const uint32_t nb01 = src0 ? src0->nb[1] : 0;
            const uint32_t nb02 = src0 ? src0->nb[2] : 0;
            const uint32_t nb03 = src0 ? src0->nb[3] : 0;

            const int32_t ne10 = src1 ? src1->ne[0] : 0;
            const int32_t ne11 = src1 ? src1->ne[1] : 0;
            const int32_t ne12 = src1 ? src1->ne[2] : 0;
//            const int32_t ne13 = src1 ? src1->ne[3] : 0;

//            const uint32_t nb10 = src1 ? src1->nb[0] : 0;
            const uint32_t nb11 = src1 ? src1->nb[1] : 0;
            const uint32_t nb12 = src1 ? src1->nb[2] : 0;
//            const uint32_t nb13 = src1 ? src1->nb[3] : 0;

            const int32_t ne0 = dst ? dst->ne[0] : 0;
            const int32_t ne1 = dst ? dst->ne[1] : 0;
            const int32_t ne2 = dst ? dst->ne[2] : 0;
//            const int32_t ne3 = dst ? dst->ne[3] : 0;

            const uint32_t nb0 = dst ? dst->nb[0] : 0;
            const uint32_t nb1 = dst ? dst->nb[1] : 0;
            const uint32_t nb2 = dst ? dst->nb[2] : 0;
            const uint32_t nb3 = dst ? dst->nb[3] : 0;

            const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
            const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
            const enum ggml_type dstt = dst ? dst->type : GGML_TYPE_COUNT;

            const static std::shared_ptr<kp::Tensor> nullTensor = nullptr;
            uint32_t off_src0 = 0;
            uint32_t off_src1 = 0;
            uint32_t off_dst = 0;
            const std::shared_ptr<kp::Tensor>& id_src0 = src0 ? ggml_vk_get_tensor(ctx, src0, &off_src0) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_src1 = src1 ? ggml_vk_get_tensor(ctx, src1, &off_src1) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_dst  = dst ? ggml_vk_get_tensor(ctx, dst, &off_dst)  : nullTensor;

            switch (dst->op) {
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_PERMUTE:
                    {
                        // noop
                    } break;
                case GGML_OP_ADD:
                    {
                        if (ggml_nelements(src1) == ne10) {
                            // src1 is a row
                            ggml_vk_addrow(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ggml_nelements(dst)/4, ne00);
                        } else {
                            ggml_vk_add(seq, id_src0, id_src1, id_dst,  off_src0, off_src1, off_dst, ggml_nelements(dst)/4);
                        }
                    } break;
                case GGML_OP_MUL:
                    {
                        if (ggml_nelements(src1) == ne10) {
                            // src1 is a row
                            ggml_vk_mulrow(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ggml_nelements(dst)/4, ne00);
                        } else {
                            ggml_vk_mul(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ggml_nelements(dst)/4);
                        }
                    } break;
                case GGML_OP_SCALE:
                    {
                        const float scale = *(const float *) src1->data;
                        ggml_vk_scale(seq, id_src0, id_dst, off_src0, off_dst, ggml_nelements(dst)/8, scale);
                    } break;
                case GGML_OP_UNARY:
                    switch (ggml_get_unary_op(gf->nodes[i])) {
                        case GGML_UNARY_OP_SILU:
                            {
                                ggml_vk_silu(seq, id_src0, id_dst, off_src0, off_dst, ggml_nelements(dst)/4);
                            } break;
                        case GGML_UNARY_OP_RELU:
                            {
                                ggml_vk_relu(seq, id_src0, id_dst, off_src0, off_dst, ggml_nelements(dst)/4);
                            } break;
                        case GGML_UNARY_OP_GELU:
                            {
                                ggml_vk_gelu(seq, id_src0, id_dst, off_src0, off_dst, ggml_nelements(dst)/8);
                            } break;
                        default:
                            {
                                fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                GGML_ASSERT(false);
                            }
                    } break;
                case GGML_OP_SOFT_MAX:
                    {
                        ggml_vk_soft_max(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03);
                    } break;
                case GGML_OP_DIAG_MASK_INF:
                    {
                        const int n_past = ((int32_t *)(dst->op_params))[0];
                        ggml_vk_diag_mask_inf(seq, id_src0, id_dst, off_src0, off_dst, n_past, ne00, ne01, ne02);
                    } break;
                case GGML_OP_NORM:
                    {
                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));
                        ggml_vk_norm(seq, id_src0, id_dst, off_src0, off_dst, ne00, nb01, ggml_nrows(src0), eps);
                    } break;
                case GGML_OP_RMS_NORM:
                    {
                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));
                        ggml_vk_rms_norm(seq, id_src0, id_dst, off_src0, off_dst, ne00, nb01, ggml_nrows(src0), eps);
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        if (src1t != GGML_TYPE_F32) {
                            fprintf(stderr, "%s: %s: Unsupported src1 type: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                            goto not_implemented;
                        }

                        if (ggml_is_transposed(src0) ||
                            ggml_is_transposed(src1)) {
                            fprintf(stderr, "%s: %s: matmul on tranposed tensor not supported: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                            goto not_implemented;
                        }

                        switch (src0t) {
                            case GGML_TYPE_F32:
                                ggml_vk_mul_mat_mat_f32(seq,
                                        id_src0, id_src1, id_dst,
                                        off_src0, off_src1, off_dst,
                                        ne00, ne01, ne02,
                                        nb01, nb02,
                                        ne11, ne12,
                                        nb11, nb12,
                                        nb1, nb2);
                                break;
                            case GGML_TYPE_F16:
                                ggml_vk_mul_mat_f16(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne01, ne02, nb01, nb02, ne11, ne12, nb11, nb12, ne0, ne1);
                                break;
                            case GGML_TYPE_Q8_0:
                                ggml_vk_mul_mat_q8_0(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne01, nb01, nb02, ne11, ne12, nb11, nb12, ne0, ne1);
                                break;
                            case GGML_TYPE_Q4_0:
                                ggml_vk_mul_mat_q4_0(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne10, ne0, ne1, ne01, ne11, ne12, ne02);
                                break;
                            case GGML_TYPE_Q4_1:
                                ggml_vk_mul_mat_q4_1(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne10, ne0, ne1, ne01, ne11, ne12, ne02);
                                break;
                            case GGML_TYPE_Q6_K:
                                ggml_vk_mul_mat_q6_k(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne10, ne0, ne1, ne01, ne11, ne12, ne02);
                                break;
                            default: {
                                fprintf(stderr, "%s: %s: Unsupported quantization: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                                goto not_implemented;
                            }
                        }

                    } break;
                case GGML_OP_GET_ROWS:
                    {
                        if (src0t == GGML_TYPE_F16) {
                            ggml_vk_get_rows_f16(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q4_0) {
                            ggml_vk_get_rows_q4_0(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q4_1) {
                            ggml_vk_get_rows_q4_1(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q6_K) {
                            ggml_vk_get_rows_q6_k(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else {
                            fprintf(stderr, "%s: %s: Unsupported quantization: %u\n", __func__, ggml_op_name(dst->op), src0t);
                            goto not_implemented;
                        }
                    } break;
                case GGML_OP_ROPE:
                    {
                        const int n_past = ((int32_t *) dst->op_params)[0];
                        const int n_dims = ((int32_t *) dst->op_params)[1];
                        const int mode   = ((int32_t *) dst->op_params)[2];
                        float freq_base;
                        float freq_scale;
                        memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
                        memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));
                        ggml_vk_rope(seq, id_src0, id_dst, off_src0, off_dst, n_past, n_dims, mode, freq_base, freq_scale, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, nb0, nb1, nb2, nb3);
                    } break;
                case GGML_OP_DUP:
                case GGML_OP_CPY:
                case GGML_OP_CONT:
                    {
                        switch (src0t) {
                            case GGML_TYPE_F32:
                                {
                                    switch (dstt) {
                                        case GGML_TYPE_F16: ggml_vk_cpy_f32_f16(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        case GGML_TYPE_F32: ggml_vk_cpy_f32_f32(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        default: goto not_implemented;
                                    }
                                } break;
                            case GGML_TYPE_F16:
                                {
                                    switch (dstt) {
                                        case GGML_TYPE_F16: ggml_vk_cpy_f16_f16(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        case GGML_TYPE_F32: ggml_vk_cpy_f16_f32(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                    default: goto not_implemented;
                                } break;
                            default: goto not_implemented;
                            }
                        }
                    } break;
                default: goto not_implemented;
            }
            continue;
            not_implemented: {}
            fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
            //GGML_ASSERT(false);
        }

        // Evaluate sequence
        seq.evalAsync();
    }

    // Wait for all sequences to finish
    for (auto& sequence : sequences) {
        if (sequence->isRunning())
            sequence->evalAwait();
    }

    ggml_vk_free_descriptor_pool(ctx);
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<half>::dataType()
{
    return TensorDataTypes::eFloat;
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<uint8_t>::dataType()
{
    return TensorDataTypes::eUnsignedInt;
}
