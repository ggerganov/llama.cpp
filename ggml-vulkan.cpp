#include "ggml-vulkan.h"

#ifdef VK_CHK_KERNEL
#include <cblas.h>
#include <cmath>
#include <chrono>
#endif

#include <vulkan/vulkan.hpp>
#define VMA_IMPLEMENTATION
#if UINTPTR_MAX == 0xFFFFFFFF
    #define VMA_SYSTEM_MEM_ALIGN 4
#else
    #define VMA_SYSTEM_MEM_ALIGN 16
#endif
#if defined(_MSC_VER) || defined(__MINGW32__)
#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment)  _aligned_malloc(size, alignment)
#define VMA_SYSTEM_ALIGNED_FREE(ptr)     _aligned_free(ptr)
#else
inline static void* ggml_aligned_malloc(size_t size, size_t alignment) {
    void* aligned_memory = NULL;
    int result = posix_memalign(&aligned_memory, alignment >= 8 ? alignment : 8, size);
    if (result != 0) {
        // Handle allocation failure
        return NULL;
    }
    return aligned_memory;
}
#define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment)  ggml_aligned_malloc(size, alignment)
#define VMA_SYSTEM_ALIGNED_FREE(ptr)     free(ptr)
#endif
#include "external/vk_mem_alloc.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "ggml.h"

#define VK_API_VERSION VK_API_VERSION_1_2

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

struct vk_buffer {
    vk::Buffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
    size_t size = 0;
    // Staging buffers
    vk_buffer * sb_write;
    vk_buffer * sb_read;
};

struct vk_pipeline {
    vk::DescriptorSetLayout dsl;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
};

vk::Instance vk_instance;
uint32_t vk_compute_queue_family_index;
uint32_t vk_transfer_queue_family_index;
vk::PhysicalDevice vk_physical_device;
vk::Device vk_device;
vk::CommandPool vk_command_pool_compute, vk_command_pool_transfer;
VmaAllocator vk_allocator;
vk_pipeline vk_pipeline_matmul_f32, vk_pipeline_matmul_f16;
VmaAllocation vk_buffer_qa_alloc, vk_buffer_a_alloc, vk_buffer_b_alloc, vk_buffer_c_alloc;
vk::Buffer vk_buffer_qa, vk_buffer_a, vk_buffer_b, vk_buffer_c;

bool vk_fp16_support = false;

static std::vector<std::tuple<void*, size_t, vk_buffer>> vk_buf_list;

static vk_pipeline ggml_vk_create_pipeline(const std::string& path, const std::string& entrypoint, const std::vector<vk::DescriptorSetLayoutBinding>& dsl_binding, const vk::PushConstantRange& pcr) {
    vk_pipeline pipeline;

    std::vector<char> matmul_shader_contents;
    if (std::ifstream shader_file{ path, std::ios::binary | std::ios::ate }) {
        const size_t file_size = shader_file.tellg();
        shader_file.seekg(0);
        matmul_shader_contents.resize(file_size, '\0');
        shader_file.read(matmul_shader_contents.data(), file_size);
    } else {
        std::cerr << "ggml_vulkan: Invalid shader path " << path << std::endl;
        abort();
    }

    vk::ShaderModuleCreateInfo shader_module_create_info(
        vk::ShaderModuleCreateFlags(),
        matmul_shader_contents.size(),
        reinterpret_cast<const uint32_t*>(matmul_shader_contents.data())
    );
    vk::ShaderModule shader_module = vk_device.createShaderModule(shader_module_create_info);

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
        vk::DescriptorSetLayoutCreateFlags(),
        dsl_binding);
    pipeline.dsl = vk_device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), pipeline.dsl, pcr);
    pipeline.layout = vk_device.createPipelineLayout(pipeline_layout_create_info);
    vk::PipelineCache pipeline_cache = vk_device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shader_module,
            entrypoint.c_str());
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),
        pipeline_shader_create_info,
        pipeline.layout);
    pipeline.pipeline = vk_device.createComputePipeline(pipeline_cache, compute_pipeline_create_info).value;

    return pipeline;
}

void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k);
void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k);

void ggml_vk_init(void) {
    char* GGML_VULKAN_DEVICE = getenv("GGML_VULKAN_DEVICE");
    int dev_num = (GGML_VULKAN_DEVICE == NULL ? 0 : atoi(GGML_VULKAN_DEVICE));

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION };
    const std::vector<const char*> layers = {
#ifdef GGML_DEBUG
        "VK_LAYER_KHRONOS_validation",
#endif
    };
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers.size(), layers.data());
    vk_instance = vk::createInstance(instance_create_info);

    vk_physical_device = vk_instance.enumeratePhysicalDevices()[dev_num];
    vk::PhysicalDeviceProperties device_props = vk_physical_device.getProperties();
    std::cout << "ggml_vulkan: Using " << device_props.deviceName << std::endl;

    std::vector<vk::QueueFamilyProperties> queue_family_props = vk_physical_device.getQueueFamilyProperties();

    const size_t qfsize = queue_family_props.size();

    // Try to find a non-graphics compute queue and a transfer-focused queue
    vk_compute_queue_family_index = qfsize;
    vk_transfer_queue_family_index = qfsize;
    for (size_t i = 0; i < qfsize; i++) {
        // std::cout << i << ": " << "compute=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eCompute) << " transfer=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eTransfer) << " graphics=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics) << " protected=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eProtected) << " optical_flow_nv=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eOpticalFlowNV) << " sparse binding=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eSparseBinding)  << " video decode=" << bool(queue_family_props[i].queueFlags & vk::QueueFlagBits::eVideoDecodeKHR) << std::endl;
        if (vk_compute_queue_family_index >= qfsize && !(queue_family_props[i].queueFlags & vk::QueueFlagBits::eGraphics) && queue_family_props[i].queueFlags & vk::QueueFlagBits::eCompute) {
            vk_compute_queue_family_index = i;
        }
        if (vk_transfer_queue_family_index >= qfsize && !(queue_family_props[i].queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eVideoDecodeKHR | vk::QueueFlagBits::eProtected | vk::QueueFlagBits::eOpticalFlowNV)) && queue_family_props[i].queueFlags & vk::QueueFlagBits::eTransfer) {
            vk_transfer_queue_family_index = i;
        }
    }

    // Fall back to graphics and compute queue if not yet found
    if (vk_compute_queue_family_index == qfsize) {
        for (size_t i = 0; i < qfsize; i++) {
            if (vk_compute_queue_family_index >= qfsize && queue_family_props[i].queueFlags & vk::QueueFlagBits::eCompute) {
                vk_compute_queue_family_index = i;
            }
        }
    }

    if (vk_compute_queue_family_index == qfsize) {
        std::cerr << "ggml_vulkan: vk_compute_queue_family_index invalid" << std::endl;
        abort();
    }
    if (vk_transfer_queue_family_index == qfsize) {
        std::cerr << "ggml_vulkan: vk_transfer_queue_family_index invalid" << std::endl;
        abort();
    }

    const float compute_queue_priority = 1.0f;
    const float transfer_queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo device_queue_create_infos[] = {
        {vk::DeviceQueueCreateFlags(), vk_compute_queue_family_index, 1, &compute_queue_priority},
        {vk::DeviceQueueCreateFlags(), vk_transfer_queue_family_index, 1, &transfer_queue_priority},
    };
    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), device_queue_create_infos);
    vk_device = vk_physical_device.createDevice(device_create_info);

    // Allocator
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.vulkanApiVersion = VK_API_VERSION;
    allocator_info.physicalDevice = vk_physical_device;
    allocator_info.device = vk_device;
    allocator_info.instance = vk_instance;

    vmaCreateAllocator(&allocator_info, &vk_allocator);

    // Shaders
    std::vector<vk::DescriptorSetLayoutBinding> dsl_binding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        6 * sizeof(int)
    );
    vk_pipeline_matmul_f32 = ggml_vk_create_pipeline("vk_shaders/matmul_f32.spv", "main", dsl_binding, pcr);
    vk_pipeline_matmul_f16 = ggml_vk_create_pipeline("vk_shaders/matmul_f16.spv", "main", dsl_binding, pcr);

    // Command pools
    vk::CommandPoolCreateInfo command_pool_create_info_compute(vk::CommandPoolCreateFlags(), vk_compute_queue_family_index);
    vk_command_pool_compute = vk_device.createCommandPool(command_pool_create_info_compute);

    vk::CommandPoolCreateInfo command_pool_create_info_transfer(vk::CommandPoolCreateFlags(), vk_transfer_queue_family_index);
    vk_command_pool_transfer = vk_device.createCommandPool(command_pool_create_info_transfer);

    // for (size_t m = 1; m < 10; m++) {
    //     for (size_t n = 1; n < 10; n++) {
    //         for (size_t k = 1; k < 10; k++) {
    //             ggml_vk_test_matmul_f32(m * 128, n * 128, k * 128);
    //             ggml_vk_test_matmul_f16(m * 128, n * 128, k * 128);
    //         }
    //     }
    // }
}

// buffer pool for vulkan
#define MAX_VK_BUFFERS 256

struct scoped_spin_lock {
    std::atomic_flag& lock;
    scoped_spin_lock(std::atomic_flag& lock) : lock(lock) {
        while (lock.test_and_set(std::memory_order_acquire)) {
            ; // spin
        }
    }
    ~scoped_spin_lock() {
        lock.clear(std::memory_order_release);
    }
    scoped_spin_lock(const scoped_spin_lock&) = delete;
    scoped_spin_lock& operator=(const scoped_spin_lock&) = delete;
};

static vk_buffer g_vk_buffer_pool[MAX_VK_BUFFERS];
static std::atomic_flag g_vk_pool_lock = ATOMIC_FLAG_INIT;

static vk_buffer ggml_vk_create_buffer(size_t size, VmaAllocationCreateFlags alloc_flags, VmaMemoryUsage vma_usage, VkMemoryPropertyFlags req_flags = 0) {
    vk_buffer buf;

    buf.size = size;
    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        1,
        &vk_compute_queue_family_index
    };

    VmaAllocationCreateInfo allocation_info = {};
    allocation_info.requiredFlags = req_flags;
    allocation_info.flags = alloc_flags;
    allocation_info.usage = vma_usage;

    vmaCreateBuffer(vk_allocator,
                    (VkBufferCreateInfo*)&buffer_create_info,
                    &allocation_info,
                    (VkBuffer*)&buf.buffer,
                    &buf.allocation,
                    &buf.info);

    buf.sb_write = nullptr;
    buf.sb_read = nullptr;

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    buf.size = 0;
    vmaDestroyBuffer(vk_allocator, buf.buffer, buf.allocation);

    // Cleanup staging buffers
    if (buf.sb_write != nullptr) {
        vmaDestroyBuffer(vk_allocator, buf.sb_write->buffer, buf.sb_write->allocation);
        free(buf.sb_write);
        buf.sb_write = nullptr;
    }
    if (buf.sb_read != nullptr) {
        vmaDestroyBuffer(vk_allocator, buf.sb_read->buffer, buf.sb_read->allocation);
        free(buf.sb_read);
        buf.sb_read = nullptr;
    }
}

static void ggml_vk_pool_malloc(size_t size, vk_buffer* buf, VmaAllocationCreateFlags alloc_flags) {
    scoped_spin_lock lock(g_vk_pool_lock);

    int best_i = -1;
    size_t best_size = std::numeric_limits<size_t>::max(); //smallest unused buffer that fits our needs
    int worst_i = -1;
    size_t worst_size = 0; //largest unused buffer seen so far
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer &b = g_vk_buffer_pool[i];
        if (b.size > 0 && b.size >= size && b.size < best_size) {
            best_i = i;
            best_size = b.size;
        }
        if (b.size > 0 && b.size > worst_size) {
            worst_i = i;
            worst_size = b.size;
        }
    }
    if(best_i != -1) {
        //found the smallest buffer that fits our needs
        vk_buffer& b = g_vk_buffer_pool[best_i];
        *buf = b;
        b.size = 0;
        return;
    }
    if(worst_i != -1) {
        //no buffer that fits our needs, resize largest one to save memory
        vk_buffer& b = g_vk_buffer_pool[worst_i];
        ggml_vk_destroy_buffer(b);
    }

    *buf = ggml_vk_create_buffer(size, alloc_flags, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
}

static void ggml_vk_pool_free(vk_buffer& buffer) {
    scoped_spin_lock lock(g_vk_pool_lock);

    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer& b = g_vk_buffer_pool[i];
        if (b.size == 0) {
            b = buffer;
            return;
        }
    }
    fprintf(stderr, "WARNING: vk buffer pool full, increase MAX_VK_BUFFERS\n");
    ggml_vk_destroy_buffer(buffer);
}

void* ggml_vk_host_malloc(size_t size) {
    if (getenv("GGML_VK_NO_PINNED") != nullptr) {
        return nullptr;
    }

    vk_buffer buf = ggml_vk_create_buffer(size, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, buf.allocation, &mem_prop_flags);

    if(!(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        buf.size = 0;
        vmaDestroyBuffer(vk_allocator, buf.buffer, buf.allocation);
        return nullptr;
    }

    vk_buf_list.push_back(std::make_tuple(buf.info.pMappedData, size, buf));

    return buf.info.pMappedData;
}

void ggml_vk_host_free(void* ptr) {
    vk_buffer* buf = nullptr;
    for (size_t i = 0; i < vk_buf_list.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
        const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
        if (ptr >= addr && ptr < endr) {
            buf = &std::get<2>(vk_buf_list[i]);
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(*buf);
}

static vk::CommandBuffer ggml_vk_cmd_buffer_create(vk::CommandPool& pool) {
    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = vk_device.allocateCommandBuffers(command_buffer_alloc_info);
    return cmd_buffers.front();
}

static void ggml_vk_buffer_write(vk_buffer* dst, size_t offset, const void * src, size_t size) {
    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, dst->allocation, &mem_prop_flags);

    // Buffer is already mapped
    if(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        memcpy((uint8_t *)dst->info.pMappedData + offset, src, size);
        if (!(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            vmaFlushAllocation(vk_allocator, dst->allocation, offset, size);
        }
    } else {
        // Check if src is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        for (size_t i = 0; i < vk_buf_list.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
            const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
            if (src >= addr && src < endr) {
                buf = &std::get<2>(vk_buf_list[i]);
                buf_offset = ((const uint8_t *)src) - addr;
                break;
            }
        }

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            VkBufferCopy buf_copy = {
                buf_offset, // srcOffset
                offset, // dstOffset,
                size}; // size

            vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_transfer);
            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            vkCmdCopyBuffer(cmd_buffer, buf->buffer, dst->buffer, 1, &buf_copy);
            cmd_buffer.end();

            vk::Queue queue = vk_device.getQueue(vk_transfer_queue_family_index, 0);

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);
            queue.submit({ submit_info }, VK_NULL_HANDLE);
            return;
        }

        // Staging buffer required, malloc because of async transfer
        if (dst->sb_write == nullptr) {
            dst->sb_write = (vk_buffer *) malloc(sizeof(vk_buffer));
            *dst->sb_write = ggml_vk_create_buffer(size, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO, 0);
        }

        memcpy(((uint8_t *)dst->sb_write->info.pMappedData) + offset, src, size);
        vmaFlushAllocation(vk_allocator, dst->sb_write->allocation, 0, VK_WHOLE_SIZE);
        VkBufferCopy buf_copy = {
            0, // srcOffset
            offset, // dstOffset,
            size}; // size

        vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_transfer);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        vkCmdCopyBuffer(cmd_buffer, dst->sb_write->buffer, dst->buffer, 1, &buf_copy);
        cmd_buffer.end();

        vk::Queue queue = vk_device.getQueue(vk_transfer_queue_family_index, 0);

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        queue.submit({ submit_info }, VK_NULL_HANDLE);
    }
}

static void ggml_vk_buffer_read(vk_buffer* src, size_t offset, void * dst, size_t size) {
    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, src->allocation, &mem_prop_flags);

    if(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (!(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            vmaInvalidateAllocation(vk_allocator, src->allocation, offset, size);
        }
        memcpy(dst, (uint8_t *) src->info.pMappedData + offset, size);
    } else {
        // Check if dst is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        for (size_t i = 0; i < vk_buf_list.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
            const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
            if (dst >= addr && dst < endr) {
                buf = &std::get<2>(vk_buf_list[i]);
                buf_offset = ((const uint8_t *)dst) - addr;
                break;
            }
        }

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            VkBufferCopy buf_copy = {
                offset, // srcOffset
                buf_offset, // dstOffset,
                size}; // size

            vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_transfer);
            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            vkCmdCopyBuffer(cmd_buffer, src->buffer, buf->buffer, 1, &buf_copy);
            cmd_buffer.end();

            vk::Queue queue = vk_device.getQueue(vk_transfer_queue_family_index, 0);
            vk::Fence fence = vk_device.createFence(vk::FenceCreateFlags{});

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);
            queue.submit({ submit_info }, fence);
            vk_device.waitForFences({ fence },
                                    true,
                                    uint64_t(-1));

            vk_device.destroyFence(fence);
            return;
        }

        if (src->sb_read == nullptr) {
            src->sb_read = (vk_buffer *) malloc(sizeof(vk_buffer));
            *src->sb_read = ggml_vk_create_buffer(size, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO, 0);
        }

        VkBufferCopy buf_copy = {
            offset, // srcOffset
            0, // dstOffset,
            size}; // size
        vmaInvalidateAllocation(vk_allocator, src->sb_read->allocation, 0, VK_WHOLE_SIZE);

        vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_transfer);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        vkCmdCopyBuffer(cmd_buffer, src->buffer, src->sb_read->buffer, 1, &buf_copy);
        cmd_buffer.end();

        vk::Queue queue = vk_device.getQueue(vk_transfer_queue_family_index, 0);
        vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo{});

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        queue.submit({ submit_info }, fence);
        vk_device.waitForFences({ fence },
                                true,
                                uint64_t(-1));
        memcpy(dst, src->sb_read->info.pMappedData, size);

        vk_device.destroyFence(fence);
        ggml_vk_destroy_buffer(*src->sb_read);
        src->sb_read = nullptr;
    }
}

static void ggml_vk_h2d_tensor_2d(vk_buffer* dst, size_t offset, const struct ggml_tensor * src, uint64_t i3, uint64_t i2) {
    const uint64_t ne0 = src->ne[0];
    const uint64_t ne1 = src->ne[1];
    const uint64_t nb0 = src->nb[0];
    const uint64_t nb1 = src->nb[1];
    const uint64_t nb2 = src->nb[2];
    const uint64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const size_t ts = ggml_type_size(type);
    const size_t bs = ggml_blck_size(type);

    const void * x = (const void *) ((const char *) src->data + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        ggml_vk_buffer_write(dst, offset, x, ne1*nb1);
        return;
    }
    if (nb0 == ts) {
        for (uint64_t i1 = 0; i1 < ne1; i1++) {
            ggml_vk_buffer_write(dst, offset + ne0 * i1, (uint8_t *)x + ts*ne0/bs, ne0*nb0);
        }
        return;
    }
    uint8_t* dst_ptr = (uint8_t*) dst->info.pMappedData;
    uint8_t* xc = (uint8_t*)x;
    for (uint64_t i1 = 0; i1 < ne1; i1++) {
        for (uint64_t i0 = 0; i0 < ne0; i0++) {
            dst_ptr[offset + i1 * ts*ne0/bs + i0 * ts] = xc[i1 * nb1 + i0 * nb0];
        }
    }
}

static void ggml_vk_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    if (src0->backend == GGML_BACKEND_GPU) {
        d_X = *(vk_buffer*) src0->data;
    } else {
        ggml_vk_pool_malloc(ggml_type_size(src0->type) * x_ne, &d_X, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    }
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 3);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = vk_device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1, &vk_pipeline_matmul_f32.dsl);
    const std::vector<vk::DescriptorSet> descriptor_sets = vk_device.allocateDescriptorSets(descriptor_set_alloc_info);
    vk::DescriptorSet descriptor_set = descriptor_sets.front();
    vk::DescriptorBufferInfo d_X_buffer_info(d_X.buffer, 0, sizeof(float) * x_ne);
    vk::DescriptorBufferInfo d_Y_buffer_info(d_Y.buffer, 0, sizeof(float) * y_ne);
    vk::DescriptorBufferInfo d_D_buffer_info(d_D.buffer, 0, sizeof(float) * d_ne);

    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets = {
        {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_X_buffer_info},
        {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_Y_buffer_info},
        {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_D_buffer_info},
    };
    vk_device.updateDescriptorSets(write_descriptor_sets, {});

    std::array<int, 6> push_constants = { (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01 };
    assert( ( sizeof( push_constants ) <= vk_physical_device.getProperties().limits.maxPushConstantsSize ) && "Too many push constants" );

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_compute);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy data to device
            if (src0->backend != GGML_BACKEND_GPU) {
                ggml_vk_h2d_tensor_2d(&d_X, 0, src0, i03, i02);
            }
            ggml_vk_h2d_tensor_2d(&d_Y, 0, src1, i03, i02);

            // compute
#ifdef VK_CHK_KERNEL
            auto begin = std::chrono::high_resolution_clock::now();
#endif

            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            cmd_buffer.pushConstants<int>(vk_pipeline_matmul_f32.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, vk_pipeline_matmul_f32.pipeline);
            cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                         vk_pipeline_matmul_f32.layout,
                                         0,
                                         { descriptor_set },
                                         {});
            cmd_buffer.dispatch(CEIL_DIV(ne01, 128), CEIL_DIV(ne11, 128), 1);
            cmd_buffer.end();

            vk::Queue queue = vk_device.getQueue(vk_compute_queue_family_index, 0);

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);

            // Wait for transfers to finish
            vk_device.getQueue(vk_transfer_queue_family_index, 0).waitIdle();

            queue.submit({ submit_info }, fence);
            vk_device.waitForFences({ fence },
                                    true,
                                    uint64_t(-1));

#ifdef VK_CHK_KERNEL
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "m=" << ne01 << " n=" << ne11 << " k=" << ne10 << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms" << std::endl;
#endif

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne);

#ifdef VK_CHK_KERNEL
            const float * x = (float *) ((char *) src0->data);
            const float * y = (float *) ((char *) src1->data);
            float * d_chk = (float *) malloc(sizeof(float) * d_ne);

            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    ne01, ne11, ne10,
                    1.0f,    x, ne00,
                             y, ne10,
                    0.0f,    d_chk, ne01);

            for (size_t i = 0; i < d_ne; i++) {
                if (std::fabs(d[i] - d_chk[i]) > 0.01f) {
                    printf("d[%ld] = %f d_chk[%ld] = %f\n", i, d[i], i, d_chk[i]);
                    abort();
                }
            }

            free(d_chk);
#endif
        }
    }

    vk_device.destroyFence(fence);

    if (src0->backend != GGML_BACKEND_GPU) {
        ggml_vk_pool_free(d_X);
    }
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);
}

static void ggml_vk_mul_mat_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, void * wdata) {
    GGML_ASSERT(vk_fp16_support);

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    if (src0->backend == GGML_BACKEND_GPU) {
        d_X = *(vk_buffer*) src0->data;
    } else {
        ggml_vk_pool_malloc(sizeof(float) * x_ne, &d_X, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    }
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    bool src0_cont_rows = nb00 == sizeof(float);
    bool src0_cont_cols = (size_t)nb01 == ne01*sizeof(float);

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 3);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = vk_device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1, &vk_pipeline_matmul_f16.dsl);
    const std::vector<vk::DescriptorSet> descriptor_sets = vk_device.allocateDescriptorSets(descriptor_set_alloc_info);
    vk::DescriptorSet descriptor_set = descriptor_sets.front();
    vk::DescriptorBufferInfo d_X_buffer_info(d_X.buffer, 0, sizeof(float) * x_ne);
    vk::DescriptorBufferInfo d_Y_buffer_info(d_Y.buffer, 0, sizeof(float) * y_ne);
    vk::DescriptorBufferInfo d_D_buffer_info(d_D.buffer, 0, sizeof(float) * d_ne);

    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets = {
        {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_X_buffer_info},
        {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_Y_buffer_info},
        {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_D_buffer_info},
    };
    vk_device.updateDescriptorSets(write_descriptor_sets, {});

    std::array<int, 6> push_constants = { (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01 };
    assert( ( sizeof( push_constants ) <= vk_physical_device.getProperties().limits.maxPushConstantsSize ) && "Too many push constants" );

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_compute);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy data to device
            if (src1->backend != GGML_BACKEND_GPU) {
                ggml_vk_h2d_tensor_2d(&d_Y, 0, src1, i03, i02);
            }
            // convert src1 to fp16
            // TODO: use multiple threads
            float * const tmp = (float *) wdata + (ne11 * ne10) * (i03 * ne02 + i02);
            char * src0i = (char *) src0->data + i03*nb03 + i02*nb02;
            if (src0_cont_rows) {
                if (src0_cont_cols) {
                    ggml_fp16_to_fp32_row((ggml_fp16_t *) src0i, tmp, ne00*ne01);
                }
                else {
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        ggml_fp16_to_fp32_row((ggml_fp16_t *) (src0i + i01*nb01), tmp + i01*ne00, ne00);
                    }
                }
            }
            else {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        // very slow due to no inlining
                        tmp[i01*ne10 + i00] = ggml_fp16_to_fp32(*(ggml_fp16_t *) (src0i + i01*nb01 + i00*nb00));
                    }
                }
            }
            ggml_vk_buffer_write(&d_X, 0, tmp, sizeof(float) * x_ne);

            // compute
#ifdef VK_CHK_KERNEL
            auto begin = std::chrono::high_resolution_clock::now();
#endif

            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            cmd_buffer.pushConstants<int>(vk_pipeline_matmul_f32.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
            cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, vk_pipeline_matmul_f32.pipeline);
            cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                         vk_pipeline_matmul_f32.layout,
                                         0,
                                         { descriptor_set },
                                         {});
            cmd_buffer.dispatch(CEIL_DIV(ne01, 128), CEIL_DIV(ne11, 128), 1);
            cmd_buffer.end();

            vk::Queue queue = vk_device.getQueue(vk_compute_queue_family_index, 0);

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);

            // Wait for transfers to finish
            vk_device.getQueue(vk_transfer_queue_family_index, 0).waitIdle();

            queue.submit({ submit_info }, fence);
            vk_device.waitForFences({ fence },
                                    true,
                                    uint64_t(-1));

#ifdef VK_CHK_KERNEL
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "m=" << ne01 << " n=" << ne11 << " k=" << ne10 << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms" << std::endl;
#endif

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_buffer_read(&d_D, 0, tmp, sizeof(float) * d_ne);

#ifdef VK_CHK_KERNEL
            for (size_t i = 0; i < d_ne; i++) {
                if (std::fabs(tmp[i] - d[i]) > 0.01f) {
                    printf("d[%ld] = %f d_chk[%ld] = %f\n", i, tmp[i], i, d[i]);
                    abort();
                }
            }
#else
            // ggml_fp16_to_fp32_row(tmp, d, d_ne);
#endif
        }
    }

    vk_device.destroyFence(fence);

    if (src0->backend != GGML_BACKEND_GPU) {
        ggml_vk_pool_free(d_X);
    }
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);
}

static void ggml_vk_mul_mat_q_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    assert(false);
//     const int64_t ne00 = src0->ne[0];
//     const int64_t ne01 = src0->ne[1];
//     const int64_t ne02 = src0->ne[2];
//     const int64_t ne03 = src0->ne[3];
//
//     const int64_t ne10 = src1->ne[0];
//     const int64_t ne11 = src1->ne[1];
//
//     const int nb2  = dst->nb[2];
//     const int nb3  = dst->nb[3];
//     const ggml_type type = src0->type;
//     const bool mul_mat_vec = ne11 == 1;
//
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
//     const int x_ne = ne01 * ne00;
//     const int y_ne = ne11 * ne10;
//     const int d_ne = ne11 * ne01;
//     const size_t q_sz = ggml_type_size(type) * x_ne / ggml_blck_size(type);
//
//     size_t x_size;
//     size_t y_size;
//     size_t d_size;
//     size_t q_size;
//     vk_buffer d_X;
//     if (!mul_mat_vec) {
//         d_X = ggml_vk_pool_malloc(sizeof(float) * x_ne, &x_size);
//     }
//     vk_buffer d_Y = ggml_vk_pool_malloc(sizeof(float) * y_ne, &y_size);
//     vk_buffer d_D = ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_size);
//     vk_buffer d_Q;
//     if (src0->backend == GGML_BACKEND_CPU) {
//         d_Q = ggml_vk_pool_malloc(q_sz, &q_size);
//     }
//
//     vk_kernel* to_fp32_vk = ggml_get_to_fp32_vk(type);
//     vk_kernel* dmmv = ggml_get_dequantize_mul_mat_vec_vk(type);
//     GGML_ASSERT(to_fp32_vk != nullptr);
//
//     size_t ev_idx = 0;
//     std::vector<vk_event> events;
//
//     for (int64_t i03 = 0; i03 < ne03; i03++) {
//         for (int64_t i02 = 0; i02 < ne02; i02++) {
//             // copy src0 to device if necessary
//             if (src0->backend == GGML_BACKEND_CPU) {
//                 events.emplace_back();
//                 VK_CHECK(ggml_vk_h2d_tensor_2d(queue, d_Q, 0, src0, i03, i02, events.data() + ev_idx++));
//             } else if (src0->backend == GGML_BACKEND_GPU) {
//                 d_Q = (vk_buffer) src0->data;
//             } else {
//                 GGML_ASSERT(false);
//             }
//             if (mul_mat_vec) { // specialized dequantize_mul_mat_vec kernel
//                 // copy src1 to device
//                 events.emplace_back();
//                 VK_CHECK(ggml_vk_h2d_tensor_2d(queue, d_Y, 0, src1, i03, i02, events.data() + ev_idx++));
//
//                 // compute
//                 const size_t global = ne01 * VK_DMMV_BLOCK_SIZE;
//                 const size_t local = VK_DMMV_BLOCK_SIZE;
//                 const vk_int ncols = ne00;
//                 events.emplace_back();
//                 VK_CHECK(vkSetKernelArg(*dmmv, 0, sizeof(vk_buffer), &d_Q));
//                 VK_CHECK(vkSetKernelArg(*dmmv, 1, sizeof(float) * local, NULL));
//                 VK_CHECK(vkSetKernelArg(*dmmv, 2, sizeof(vk_buffer), &d_Y));
//                 VK_CHECK(vkSetKernelArg(*dmmv, 3, sizeof(vk_buffer), &d_D));
//                 VK_CHECK(vkSetKernelArg(*dmmv, 4, sizeof(vk_int), &ncols));
//                 VK_CHECK(vkEnqueueNDRangeKernel(queue, *dmmv, 1, NULL, &global, &local, events.size() - 1, events.data(), events.data() + ev_idx++));
//             } else { // general dequantization kernel + VKBlast matrix matrix multiplication
//                 // convert src0 to fp32 on device
//                 const size_t global = x_ne;
//                 VK_CHECK(vkSetKernelArg(*to_fp32_vk, 0, sizeof(vk_buffer), &d_Q));
//                 VK_CHECK(vkSetKernelArg(*to_fp32_vk, 1, sizeof(vk_buffer), &d_X));
//                 VK_CHECK(vkEnqueueNDRangeKernel(queue, *to_fp32_vk, 1, NULL, &global, NULL, events.size(), !events.empty() ? events.data() : NULL, NULL));
//
//                 // copy src1 to device
//                 VK_CHECK(ggml_vk_h2d_tensor_2d(queue, d_Y, 0, src1, i03, i02, NULL));
//
//                 events.emplace_back();
//
//                 // wait for conversion
//                 VK_CHECK(vkFinish(queue));
//
//                 // compute
//                 vkblast::StatusCode status = vkblast::Gemm<vk_float>(vkblast::Layout::kColMajor,
//                                                            vkblast::Transpose::kYes, vkblast::Transpose::kNo,
//                                                            ne01, ne11, ne10,
//                                                            alpha,
//                                                            d_X, 0, ne00,
//                                                            d_Y, 0, ne10,
//                                                            beta,
//                                                            d_D, 0, ne01,
//                                                            &queue, events.data() + ev_idx++);
//
//                 if (status != vkblast::StatusCode::kSuccess) {
//                     GGML_ASSERT(false);
//                 }
//             }
//
//             // copy dst to host
//             float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
//             VK_CHECK(vkEnqueueReadBuffer(queue, d_D, true, 0, sizeof(float) * d_ne, d, 1, &events[events.size() - 1], NULL));
//             for (auto *event : events) {
//                 vkReleaseEvent(event);
//             }
//
//             ev_idx = 0;
//             events.vkear();
//         }
//     }
//
//     if (!mul_mat_vec) {
//         ggml_vk_pool_free(d_X, x_size);
//     }
//     ggml_vk_pool_free(d_Y, y_size);
//     ggml_vk_pool_free(d_D, d_size);
//     if (src0->backend == GGML_BACKEND_CPU) {
//         ggml_vk_pool_free(d_Q, q_size);
//     }
}


bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 /*|| src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)*/) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32) || src0->backend == GGML_BACKEND_GPU)) {
        return true;
    }

    return false;
}

bool ggml_vk_mul_mat_use_f16(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * /* dst */) {
    // If device doesn't support FP16
    if (!vk_fp16_support) {
        return false;
    }

    size_t src0_sz = ggml_nbytes(src0);
    size_t src1_sz = ggml_nbytes(src1);

    // mul_mat_q: src0 is converted to fp32 on device
    size_t mul_mat_q_transfer = src0_sz + src1_sz;

    // mul_mat_f16: src1 is converted to fp16 on cpu
    size_t mul_mat_f16_transfer = src0_sz + sizeof(ggml_fp16_t) * ggml_nelements(src1);

    // choose the smaller one to transfer to the device
    // TODO: this is not always the best choice due to the overhead of converting to fp16
    return mul_mat_f16_transfer < mul_mat_q_transfer;
}

void ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize) {
    GGML_ASSERT(ggml_vk_can_mul_mat(src0, src1, dst));

    if (src0->type == GGML_TYPE_F32) {
        ggml_vk_mul_mat_f32(src0, src1, dst);
    }
    else if (src0->type == GGML_TYPE_F16) {
        if (ggml_vk_mul_mat_use_f16(src0, src1, dst)) {
            ggml_vk_mul_mat_f16(src0, src1, dst, wdata);
        }
        else {
            ggml_vk_mul_mat_q_f32(src0, src1, dst);
        }
    }
    else if (ggml_is_quantized(src0->type)) {
        ggml_vk_mul_mat_q_f32(src0, src1, dst);
    }
    else {
        GGML_ASSERT(false);
    }
}

size_t ggml_vk_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    if (ggml_vk_mul_mat_use_f16(src0, src1, dst)) {
        return ggml_nelements(src1) * sizeof(ggml_fp16_t);
    }
    return 0;
}

#ifdef VK_CHK_KERNEL
void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k) {
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(float) * x_ne, &d_X, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 3);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = vk_device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1, &vk_pipeline_matmul_f32.dsl);
    const std::vector<vk::DescriptorSet> descriptor_sets = vk_device.allocateDescriptorSets(descriptor_set_alloc_info);
    vk::DescriptorSet descriptor_set = descriptor_sets.front();
    vk::DescriptorBufferInfo d_X_buffer_info(d_X.buffer, 0, sizeof(float) * x_ne);
    vk::DescriptorBufferInfo d_Y_buffer_info(d_Y.buffer, 0, sizeof(float) * y_ne);
    vk::DescriptorBufferInfo d_D_buffer_info(d_D.buffer, 0, sizeof(float) * d_ne);

    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets = {
        {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_X_buffer_info},
        {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_Y_buffer_info},
        {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_D_buffer_info},
    };
    vk_device.updateDescriptorSets(write_descriptor_sets, {});

    std::array<int, 6> push_constants = { (int)m, (int)n, (int)k, (int)k, (int)k, (int)m };
    assert( ( sizeof( push_constants ) <= vk_physical_device.getProperties().limits.maxPushConstantsSize ) && "Too many push constants" );

    float* x = (float *) malloc(sizeof(float) * x_ne);
    float* y = (float *) malloc(sizeof(float) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = rand() / (float)RAND_MAX;
    }

    ggml_vk_buffer_write(&d_X, 0, x, sizeof(float) * x_ne);
    ggml_vk_buffer_write(&d_Y, 0, y, sizeof(float) * y_ne);

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_compute);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    // compute
    auto begin = std::chrono::high_resolution_clock::now();

    vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffer.begin(cmd_buffer_begin_info);
    cmd_buffer.pushConstants<int>(vk_pipeline_matmul_f32.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, vk_pipeline_matmul_f32.pipeline);
    cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 vk_pipeline_matmul_f32.layout,
                                 0,
                                 { descriptor_set },
                                 {});
    cmd_buffer.dispatch(CEIL_DIV(m, 128), CEIL_DIV(n, 128), 1);
    cmd_buffer.end();

    vk::Queue queue = vk_device.getQueue(vk_compute_queue_family_index, 0);

    vk::SubmitInfo submit_info(0,
                               nullptr,
                               nullptr,
                               1,
                               &cmd_buffer);

    // Wait for transfers to finish
    vk_device.getQueue(vk_transfer_queue_family_index, 0).waitIdle();

    queue.submit({ submit_info }, fence);
    vk_device.waitForFences({ fence },
                            true,
                            uint64_t(-1));

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne);

    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            m, n, k,
            1.0f,    x, k,
                     y, k,
            0.0f,    d_chk, m);

    double avg_err = 0.0;

    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            avg_err += std::fabs(d[c * m + r] - d_chk[c * m + r]);
        }
    }

    std::cout << "TEST FP32 m=" << m << " n=" << n << " k=" << k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(d_chk);

    vk_device.destroyFence(fence);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    free(x);
    free(y);
    free(d);
}

void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k) {
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * x_ne, &d_X, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * y_ne, &d_Y, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * d_ne, &d_D, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, 3);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(), 1, descriptor_pool_size);
    vk::DescriptorPool descriptor_pool = vk_device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(descriptor_pool, 1, &vk_pipeline_matmul_f32.dsl);
    const std::vector<vk::DescriptorSet> descriptor_sets = vk_device.allocateDescriptorSets(descriptor_set_alloc_info);
    vk::DescriptorSet descriptor_set = descriptor_sets.front();
    vk::DescriptorBufferInfo d_X_buffer_info(d_X.buffer, 0, sizeof(ggml_fp16_t) * x_ne);
    vk::DescriptorBufferInfo d_Y_buffer_info(d_Y.buffer, 0, sizeof(ggml_fp16_t) * y_ne);
    vk::DescriptorBufferInfo d_D_buffer_info(d_D.buffer, 0, sizeof(ggml_fp16_t) * d_ne);

    const std::vector<vk::WriteDescriptorSet> write_descriptor_sets = {
        {descriptor_set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_X_buffer_info},
        {descriptor_set, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_Y_buffer_info},
        {descriptor_set, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &d_D_buffer_info},
    };
    vk_device.updateDescriptorSets(write_descriptor_sets, {});

    std::array<int, 6> push_constants = { (int)m, (int)n, (int)k, (int)k, (int)k, (int)m };
    assert( ( sizeof( push_constants ) <= vk_physical_device.getProperties().limits.maxPushConstantsSize ) && "Too many push constants" );

    ggml_fp16_t* x = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * x_ne);
    ggml_fp16_t* y = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * y_ne);
    ggml_fp16_t* d = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }

    ggml_vk_buffer_write(&d_X, 0, x, sizeof(ggml_fp16_t) * x_ne);
    ggml_vk_buffer_write(&d_Y, 0, y, sizeof(ggml_fp16_t) * y_ne);

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_command_pool_compute);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    // compute
    auto begin = std::chrono::high_resolution_clock::now();

    vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffer.begin(cmd_buffer_begin_info);
    cmd_buffer.pushConstants<int>(vk_pipeline_matmul_f32.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, vk_pipeline_matmul_f32.pipeline);
    cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 vk_pipeline_matmul_f32.layout,
                                 0,
                                 { descriptor_set },
                                 {});
    cmd_buffer.dispatch(CEIL_DIV(m, 32), CEIL_DIV(n, 32), 1);
    cmd_buffer.end();

    vk::Queue queue = vk_device.getQueue(vk_compute_queue_family_index, 0);

    vk::SubmitInfo submit_info(0,
                               nullptr,
                               nullptr,
                               1,
                               &cmd_buffer);

    // Wait for transfers to finish
    vk_device.getQueue(vk_transfer_queue_family_index, 0).waitIdle();

    queue.submit({ submit_info }, fence);
    vk_device.waitForFences({ fence },
                            true,
                            uint64_t(-1));

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(ggml_fp16_t) * d_ne);

    float * fx = (float *) malloc(sizeof(float) * x_ne);
    float * fy = (float *) malloc(sizeof(float) * y_ne);
    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    ggml_fp16_to_fp32_row(x, fx, x_ne);
    ggml_fp16_to_fp32_row(y, fy, y_ne);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            m, n, k,
            1.0f,    fx, k,
                     fy, k,
            0.0f,    d_chk, m);

    double avg_err = 0.0;

    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            avg_err += std::fabs(ggml_fp16_to_fp32(d[c * m + r]) - d_chk[c * m + r]);
        }
    }

    std::cout << "TEST FP16 m=" << m << " n=" << n << " k=" << k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(fx);
    free(fy);
    free(d_chk);

    vk_device.destroyFence(fence);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    free(x);
    free(y);
    free(d);
}
#endif
