#include "ggml-vulkan.h"

#ifdef VK_CHK_KERNEL
#include <cblas.h>
#include <cmath>
#include <chrono>

#define PROFILE(name, block) do { \
    auto begin = std::chrono::high_resolution_clock::now(); \
    block \
    auto end = std::chrono::high_resolution_clock::now(); \
    double time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0; \
    printf("%s: %lf ms\n", name, time_taken); \
} while(0)
#else
#define PROFILE(name, block) block
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
#include <mutex>

#include "ggml.h"

#define VK_API_VERSION VK_API_VERSION_1_2

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define VK_TRANSFER_QUEUE_COUNT 2

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
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSet descriptor_set;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
};

struct vk_queue {
    vk_queue() {};
    vk_queue(const vk_queue& b) : queue_family_index(b.queue_family_index), queue(b.queue), pool(b.pool) {}

    vk_queue& operator=(const vk_queue& b) {
        if (this != &b) {
            queue_family_index = b.queue_family_index;
            queue = b.queue;
            pool = b.pool;
        }
        return *this;
    }

    uint32_t queue_family_index;
    vk::Queue queue;
    vk::CommandPool pool;
    std::vector<vk::CommandBuffer> cmd_buffers;
    std::mutex mutex;
};

vk::Instance vk_instance;
vk::PhysicalDevice vk_physical_device;
vk::Device vk_device;
vk_queue vk_compute_queue;
vk_queue vk_transfer_queues[VK_TRANSFER_QUEUE_COUNT];
VmaAllocator vk_allocator;
vk_pipeline vk_pipeline_matmul_f32, vk_pipeline_matmul_f16;
vk_pipeline vk_pipeline_f16_to_f32, vk_pipeline_dequant_q4_0;
VmaAllocation vk_buffer_qa_alloc, vk_buffer_a_alloc, vk_buffer_b_alloc, vk_buffer_c_alloc;
vk::Buffer vk_buffer_qa, vk_buffer_a, vk_buffer_b, vk_buffer_c;

bool vk_fp16_support = false;

static std::vector<std::tuple<void*, size_t, vk_buffer>> vk_buf_list;

static vk_pipeline ggml_vk_create_pipeline(const std::string& path, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_count, std::array<uint32_t, 3> wg_denoms) {
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0);

    vk_pipeline pipeline;

    pipeline.parameter_count = parameter_count;
    pipeline.push_constant_size = push_constant_count * sizeof(int);
    pipeline.wg_denoms = wg_denoms;

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

    std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
    for (uint32_t i = 0; i < parameter_count; i++) {
        dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
    }

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline.push_constant_size
    );

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
        vk::DescriptorSetLayoutCreateFlags(),
        dsl_binding);
    pipeline.dsl = vk_device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline.parameter_count);
    vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlags(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT), 1, descriptor_pool_size);
    pipeline.descriptor_pool = vk_device.createDescriptorPool(descriptor_pool_create_info);

    vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pipeline.descriptor_pool, 1, &pipeline.dsl);
    const std::vector<vk::DescriptorSet> descriptor_sets = vk_device.allocateDescriptorSets(descriptor_set_alloc_info);
    pipeline.descriptor_set = descriptor_sets.front();

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

static void ggml_vk_dispatch_pipeline(vk_pipeline& pipeline, std::vector<vk_buffer *> buffers, const std::vector<int>&& push_constants, std::array<uint32_t, 3> elements, vk::CommandBuffer& cmd_buffer, vk::Fence& fence) {
    PROFILE("ggml_vk_dispatch_pipeline",
    std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    for (uint32_t i = 0; i < pipeline.parameter_count; i++) {
        descriptor_buffer_infos.push_back({buffers[i]->buffer, 0, buffers[i]->size});
    }
    for (uint32_t i = 0; i < pipeline.parameter_count; i++) {
        write_descriptor_sets.push_back({pipeline.descriptor_set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &descriptor_buffer_infos[i]});
    }

    vk_device.updateDescriptorSets(write_descriptor_sets, {});

    vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmd_buffer.begin(cmd_buffer_begin_info);
    cmd_buffer.pushConstants<int>(pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constants);
    cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
    cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 pipeline.layout,
                                 0,
                                 { pipeline.descriptor_set },

                                 {});
    cmd_buffer.dispatch(CEIL_DIV(elements[0], pipeline.wg_denoms[0]), CEIL_DIV(elements[1], pipeline.wg_denoms[1]), CEIL_DIV(elements[2], pipeline.wg_denoms[2]));
    cmd_buffer.end();

    std::lock_guard<std::mutex> guard(vk_compute_queue.mutex);

    vk::SubmitInfo submit_info(0,
                               nullptr,
                               nullptr,
                               1,
                               &cmd_buffer);

    vk_compute_queue.queue.submit({ submit_info }, fence);
    );
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, int32_t min_num_queues) {
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cout << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static vk_queue ggml_vk_create_queue(uint32_t queue_family_index, uint32_t queue_index) {
    vk_queue q;
    q.queue_family_index = queue_family_index;

    vk::CommandPoolCreateInfo command_pool_create_info_compute(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT), queue_family_index);
    q.pool = vk_device.createCommandPool(command_pool_create_info_compute);

    q.queue = vk_device.getQueue(queue_family_index, queue_index);

    return q;
}

static vk::CommandBuffer ggml_vk_cmd_buffer_create(vk_queue& q) {
    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        q.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = vk_device.allocateCommandBuffers(command_buffer_alloc_info);
    auto buf = cmd_buffers.front();

    q.cmd_buffers.push_back(buf);

    return buf;
}

static void ggml_vk_queue_cleanup(vk_queue& q) {
    q.queue.waitIdle();
    vk_device.freeCommandBuffers(q.pool, q.cmd_buffers);
    q.cmd_buffers.clear();
}

static vk_buffer ggml_vk_create_buffer(size_t size, VmaAllocationCreateFlags alloc_flags, VmaMemoryUsage vma_usage, VkMemoryPropertyFlags req_flags = 0) {
    vk_buffer buf;

    buf.size = size;
    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        1,
        nullptr,
    };

    VmaAllocationCreateInfo allocation_info = {};
    allocation_info.requiredFlags = req_flags;
    allocation_info.flags = alloc_flags;
    allocation_info.usage = vma_usage;

    PROFILE("ggml_vk_create_buffer",
    vmaCreateBuffer(vk_allocator,
                    (VkBufferCreateInfo*)&buffer_create_info,
                    &allocation_info,
                    (VkBuffer*)&buf.buffer,
                    &buf.allocation,
                    &buf.info);
    );

    buf.sb_write = nullptr;
    buf.sb_read = nullptr;

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    buf.size = 0;
    PROFILE("ggml_vk_destroy_buffer",
    vmaDestroyBuffer(vk_allocator, buf.buffer, buf.allocation);

    // Cleanup staging buffers
    if (buf.sb_write != nullptr) {
        vmaDestroyBuffer(vk_allocator, buf.sb_write->buffer, buf.sb_write->allocation);
        delete buf.sb_write;
        buf.sb_write = nullptr;
    }
    if (buf.sb_read != nullptr) {
        vmaDestroyBuffer(vk_allocator, buf.sb_read->buffer, buf.sb_read->allocation);
        delete buf.sb_read;
        buf.sb_read = nullptr;
    }
    );
}

void ggml_vk_test_transfer(size_t ne);
void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k);
void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k);

void ggml_vk_init(void) {
    char* GGML_VULKAN_DEVICE = getenv("GGML_VULKAN_DEVICE");
    int dev_num = (GGML_VULKAN_DEVICE == NULL ? 0 : atoi(GGML_VULKAN_DEVICE));

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION };
    const std::vector<const char*> layers = {
        // "VK_LAYER_KHRONOS_validation",
    };
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers.size(), layers.data());
    vk_instance = vk::createInstance(instance_create_info);

    vk_physical_device = vk_instance.enumeratePhysicalDevices()[dev_num];
    vk::PhysicalDeviceProperties device_props = vk_physical_device.getProperties();
    std::cerr << "ggml_vulkan: Using " << device_props.deviceName << std::endl;

    std::vector<vk::ExtensionProperties> ext_props = vk_physical_device.enumerateDeviceExtensionProperties();

    bool fp16_storage = false;
    bool fp16_compute = false;

    for (auto properties : ext_props) {
        if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = true;
        } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = true;
        }
    }

    vk_fp16_support = fp16_storage && fp16_compute;

    std::vector<vk::QueueFamilyProperties> queue_family_props = vk_physical_device.getQueueFamilyProperties();

    // Try to find a non-graphics compute queue and transfer-focused queues
    uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, -1, 1);
    uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eVideoDecodeKHR | vk::QueueFlagBits::eProtected | vk::QueueFlagBits::eOpticalFlowNV, compute_queue_family_index, 2);

    const float compute_queue_priority = 1.0f;
    const float transfer_queue_priority[] = { 1.0f, 1.0f, 1.0f };
    std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
    if (compute_queue_family_index != transfer_queue_family_index) {
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, &compute_queue_priority});
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, VK_TRANSFER_QUEUE_COUNT, transfer_queue_priority});
    } else {
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, 1 + VK_TRANSFER_QUEUE_COUNT, transfer_queue_priority});
    }
    vk::DeviceCreateInfo device_create_info;
    std::vector<const char *> device_extensions;
    vk::PhysicalDeviceFeatures device_features = vk_physical_device.getFeatures();

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext = nullptr;
    device_features2.features = device_features;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.pNext = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.pNext = &vk12_features;

    vkGetPhysicalDeviceFeatures2(vk_physical_device, &device_features2);

    vk_fp16_support = vk_fp16_support && vk12_features.shaderFloat16;

    if (!vk11_features.storageBuffer16BitAccess) {
        std::cerr << "ggml_vulkan: device does not support 16-bit storage" << std::endl;
    }

    device_extensions.push_back("VK_KHR_16bit_storage");

    if (vk_fp16_support) {
        std::cerr << "ggml_vulkan: 16-bit enabled" << std::endl;
        device_extensions.push_back("VK_KHR_shader_float16_int8");
    }
    device_create_info = {
        vk::DeviceCreateFlags(),
        device_queue_create_infos,
        {},
        device_extensions
    };
    device_create_info.setPNext(&device_features2);
    vk_device = vk_physical_device.createDevice(device_create_info);

    // Allocator
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.vulkanApiVersion = VK_API_VERSION;
    allocator_info.physicalDevice = vk_physical_device;
    allocator_info.device = vk_device;
    allocator_info.instance = vk_instance;

    vmaCreateAllocator(&allocator_info, &vk_allocator);

    // Shaders
    vk_pipeline_matmul_f32 = ggml_vk_create_pipeline("vk_shaders/matmul_f32.spv", "main", 3, 6, {128, 128, 1});
    if (vk_fp16_support) {
        vk_pipeline_matmul_f16 = ggml_vk_create_pipeline("vk_shaders/matmul_f16.spv", "main", 3, 6, {128, 128, 1});
    }

    vk_pipeline_f16_to_f32 = ggml_vk_create_pipeline("vk_shaders/f16_to_f32.spv", "main", 2, 1, {32, 1, 1});
    vk_pipeline_dequant_q4_0 = ggml_vk_create_pipeline("vk_shaders/dequant_q4_0.spv", "main", 2, 1, {32, 1, 1});

    // Queues
    vk_compute_queue = ggml_vk_create_queue(compute_queue_family_index, 0);
    for (int i = 0; i < VK_TRANSFER_QUEUE_COUNT; i++) {
        vk_transfer_queues[i] = ggml_vk_create_queue(transfer_queue_family_index, i);
    }

#if defined(VK_CHK_KERNEL)
    int step = 16;
    for (size_t m = step; m < 64; m += step) {
        ggml_vk_test_transfer(1024 * 1024 * m);
    }
    const std::vector<size_t> vals {
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
        11008, 49, 4096,
        4096, 49, 11008,
        4096, 49, 4096,
        32000, 49, 4096,
    };
    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2]);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2]);
    }
#endif
}

static vk_pipeline* ggml_get_to_fp32_vk(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return &vk_pipeline_dequant_q4_0;
        // case GGML_TYPE_Q4_1:
        //     return &dequantize_row_q4_1_cl;
        // case GGML_TYPE_Q5_0:
        //     return &dequantize_row_q5_0_cl;
        // case GGML_TYPE_Q5_1:
        //     return &dequantize_row_q5_1_cl;
        // case GGML_TYPE_Q8_0:
        //     return &dequantize_row_q8_0_cl;
        case GGML_TYPE_F16:
            return &vk_pipeline_f16_to_f32;
        default:
            return nullptr;
    }
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

static void ggml_vk_pool_malloc(size_t size, vk_buffer* buf, VmaAllocationCreateFlags alloc_flags) {
    PROFILE("ggml_vk_pool_malloc",
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
    );
}

static void ggml_vk_pool_free(vk_buffer& buffer) {
    PROFILE("ggml_vk_pool_free",
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
    );
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

static void ggml_vk_buffer_write(vk_buffer* dst, size_t offset, const void * src, size_t size, vk_queue& q) {
    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, dst->allocation, &mem_prop_flags);

    // Buffer is already mapped
    if(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        GGML_ASSERT(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        PROFILE("ggml_vk_buffer_write visible",
        memcpy((uint8_t *)dst->info.pMappedData + offset, src, size);
        );
    } else {
        // Check if src is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        PROFILE("ggml_vk_buffer_write pinned check",
        for (size_t i = 0; i < vk_buf_list.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
            const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
            if (src >= addr && src < endr) {
                buf = &std::get<2>(vk_buf_list[i]);
                buf_offset = ((const uint8_t *)src) - addr;
                break;
            }
        }
        );

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            VkBufferCopy buf_copy = {
                buf_offset, // srcOffset
                offset, // dstOffset,
                size}; // size

            vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
            PROFILE("ggml_vk_buffer_write pinned write",
            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            vkCmdCopyBuffer(cmd_buffer, buf->buffer, dst->buffer, 1, &buf_copy);
            cmd_buffer.end();
            );

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);
            std::lock_guard<std::mutex> guard(q.mutex);
            q.queue.submit({ submit_info }, VK_NULL_HANDLE);

            return;
        }

        // Staging buffer required, malloc because of async transfer
        if (dst->sb_write == nullptr) {
            dst->sb_write = new vk_buffer;
            *dst->sb_write = ggml_vk_create_buffer(dst->size, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 0);
        }

        VkMemoryPropertyFlags mpf_staging;
        vmaGetAllocationMemoryProperties(vk_allocator, dst->sb_write->allocation, &mpf_staging);
        GGML_ASSERT(mpf_staging & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkBufferCopy buf_copy = {
            0, // srcOffset
            offset, // dstOffset,
            size}; // size

        PROFILE("ggml_vk_buffer_write staging",
        vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        vkCmdCopyBuffer(cmd_buffer, dst->sb_write->buffer, dst->buffer, 1, &buf_copy);
        cmd_buffer.end();

        memcpy(dst->sb_write->info.pMappedData, src, size);

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        std::lock_guard<std::mutex> guard(q.mutex);
        q.queue.submit({ submit_info }, VK_NULL_HANDLE);
        );
    }
}

static void ggml_vk_buffer_write_2d(vk_buffer* dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, vk_queue& q) {
    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, dst->allocation, &mem_prop_flags);

    // Buffer is already mapped
    if(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        GGML_ASSERT(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        PROFILE("ggml_vk_buffer_write visible",
        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->info.pMappedData + offset + i * width, (uint8_t *) src + i * spitch, width);
        }
        );
    } else {
        // Check if src is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        PROFILE("ggml_vk_buffer_write pinned check",
        for (size_t i = 0; i < vk_buf_list.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
            const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
            if (src >= addr && src < endr) {
                buf = &std::get<2>(vk_buf_list[i]);
                buf_offset = ((const uint8_t *)src) - addr;
                break;
            }
        }
        );

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            std::vector<VkBufferCopy> slices(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * width;
                slices[i].size = width;
            }

            vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
            PROFILE("ggml_vk_buffer_write pinned write",
            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            vkCmdCopyBuffer(cmd_buffer, buf->buffer, dst->buffer, height, slices.data());
            cmd_buffer.end();
            );

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);
            std::lock_guard<std::mutex> guard(q.mutex);
            q.queue.submit({ submit_info }, VK_NULL_HANDLE);

            return;
        }

        // Staging buffer required, malloc because of async transfer
        if (dst->sb_write == nullptr) {
            dst->sb_write = new vk_buffer;
            *dst->sb_write = ggml_vk_create_buffer(dst->size, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 0);
        }

        VkMemoryPropertyFlags mpf_staging;
        vmaGetAllocationMemoryProperties(vk_allocator, dst->sb_write->allocation, &mpf_staging);
        GGML_ASSERT(mpf_staging & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkBufferCopy buf_copy = {
            0,
            offset,
            width * height};

        PROFILE("ggml_vk_buffer_write staging",
        vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        vkCmdCopyBuffer(cmd_buffer, dst->sb_write->buffer, dst->buffer, 1, &buf_copy);
        cmd_buffer.end();

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->info.pMappedData + offset + i * width, (uint8_t *) src + i * spitch, width);
        }

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        std::lock_guard<std::mutex> guard(q.mutex);
        q.queue.submit({ submit_info }, VK_NULL_HANDLE);
        );
    }
}

static void ggml_vk_buffer_read(vk_buffer* src, size_t offset, void * dst, size_t size, vk_queue& q) {
    VkMemoryPropertyFlags mem_prop_flags;
    vmaGetAllocationMemoryProperties(vk_allocator, src->allocation, &mem_prop_flags);

    if(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        GGML_ASSERT(mem_prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        PROFILE("ggml_vk_buffer_read visible",
        memcpy(dst, (uint8_t *) src->info.pMappedData + offset, size);
        );
    } else {
        // Check if dst is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        PROFILE("ggml_vk_buffer_write pinned check",
        for (size_t i = 0; i < vk_buf_list.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_buf_list[i]);
            const uint8_t* endr = addr + std::get<1>(vk_buf_list[i]);
            if (dst >= addr && dst < endr) {
                buf = &std::get<2>(vk_buf_list[i]);
                buf_offset = ((const uint8_t *)dst) - addr;
                break;
            }
        }
        );

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            VkBufferCopy buf_copy = {
                offset, // srcOffset
                buf_offset, // dstOffset,
                size}; // size

            PROFILE("ggml_vk_buffer_write pinned read",
            vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
            vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            cmd_buffer.begin(cmd_buffer_begin_info);
            vkCmdCopyBuffer(cmd_buffer, src->buffer, buf->buffer, 1, &buf_copy);
            cmd_buffer.end();

            vk::SubmitInfo submit_info(0,
                                       nullptr,
                                       nullptr,
                                       1,
                                       &cmd_buffer);
            std::lock_guard<std::mutex> guard(q.mutex);
            q.queue.submit({ submit_info }, VK_NULL_HANDLE);
            );
            return;
        }

        if (src->sb_read == nullptr) {
            src->sb_read = new vk_buffer;
            *src->sb_read = ggml_vk_create_buffer(src->size, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_AUTO, 0);
        }

        VkMemoryPropertyFlags mpf_staging;
        vmaGetAllocationMemoryProperties(vk_allocator, src->sb_read->allocation, &mpf_staging);
        GGML_ASSERT(mpf_staging & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkBufferCopy buf_copy = {
            offset, // srcOffset
            0, // dstOffset,
            size}; // size

        PROFILE("ggml_vk_buffer_write staging",
        vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(q);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        vkCmdCopyBuffer(cmd_buffer, src->buffer, src->sb_read->buffer, 1, &buf_copy);
        cmd_buffer.end();

        vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo{});

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        std::lock_guard<std::mutex> guard(q.mutex);
        q.queue.submit({ submit_info }, fence);
        vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "vk_buffer_read staging waitForFences");
        vk_device.destroyFence(fence);
        memcpy(dst, src->sb_read->info.pMappedData, size);
        );
    }
}

static void ggml_vk_h2d_tensor_2d(vk_buffer* dst, size_t offset, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, vk_queue& q) {
    const uint64_t ne0 = src->ne[0];
    const uint64_t ne1 = src->ne[1];
    const uint64_t nb0 = src->nb[0];
    const uint64_t nb1 = src->nb[1];
    const uint64_t nb2 = src->nb[2];
    const uint64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const size_t ts = ggml_type_size(type);
    const size_t bs = ggml_blck_size(type);
    const size_t row_length = ts*ne0/bs;

    const void * x = (const void *) ((const char *) src->data + i2*nb2 + i3*nb3);
    if (nb0 == ts && nb1 == row_length) {
        ggml_vk_buffer_write(dst, offset, x, ne1*nb1, q);
        return;
    }
    if (nb0 == ts) {
        PROFILE("ggml_vk_buffer_write_2d",
        // for (uint64_t i1 = 0; i1 < ne1; i1++) {
        //     ggml_vk_buffer_write(dst, offset + i1 * row_length, (uint8_t *)x + i1 * nb1, row_length, q);
        // }
        ggml_vk_buffer_write_2d(dst, offset, x, nb1, row_length, ne1, q);
        );
        return;
    }
    GGML_ASSERT(false);
    // TODO: also needs handling of staging buffers
    uint8_t* dst_ptr = (uint8_t*) dst->info.pMappedData;
    uint8_t* xc = (uint8_t*)x;
    for (uint64_t i1 = 0; i1 < ne1; i1++) {
        for (uint64_t i0 = 0; i0 < ne0; i0++) {
            dst_ptr[offset + i1 * row_length + i0 * ts] = xc[i1 * nb1 + i0 * nb0];
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
        ggml_vk_pool_malloc(ggml_type_size(src0->type) * x_ne, &d_X, 0);
    }
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, 0);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, 0);

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_compute_queue);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy data to device
            if (src0->backend != GGML_BACKEND_GPU) {
                ggml_vk_h2d_tensor_2d(&d_X, 0, src0, i03, i02, vk_transfer_queues[0]);
            }
            ggml_vk_h2d_tensor_2d(&d_Y, 0, src1, i03, i02, vk_transfer_queues[1]);

            // compute
#ifdef VK_CHK_KERNEL
            auto begin = std::chrono::high_resolution_clock::now();
#endif

            // Wait for transfers to finish
            vk_transfer_queues[0].queue.waitIdle();
            vk_transfer_queues[1].queue.waitIdle();

            ggml_vk_dispatch_pipeline(vk_pipeline_matmul_f32, {&d_X, &d_Y, &d_D}, { (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01 }, { (uint32_t)ne01, (uint32_t)ne11, 1}, cmd_buffer, fence);

            vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "matmul_f32 waitForFences");

#ifdef VK_CHK_KERNEL
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "m=" << ne01 << " n=" << ne11 << " k=" << ne10 << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms" << std::endl;
#endif

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_transfer_queues[0]);

            vk_device.resetFences({fence});
        }
    }

    vk_device.destroyFence(fence);

    ggml_vk_queue_cleanup(vk_compute_queue);
    ggml_vk_queue_cleanup(vk_transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_transfer_queues[1]);

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

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

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
        ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * x_ne, &d_X, 0);
    }
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * y_ne, &d_Y, 0);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, 0);

    bool src1_cont_rows = nb10 == sizeof(float);
    bool src1_cont_cols = (size_t)nb11 == ne11*sizeof(float);

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_compute_queue);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateInfo());

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy data to device
            if (src1->backend != GGML_BACKEND_GPU) {
                ggml_vk_h2d_tensor_2d(&d_X, 0, src0, i03, i02, vk_transfer_queues[0]);
            }

            // convert src1 to fp16
            // TODO: use multiple threads
            ggml_fp16_t * const tmp = (ggml_fp16_t *) wdata + (ne11 * ne10) * (i03 * ne02 + i02);
            char * src1i = (char *) src1->data + i03*nb13 + i02*nb12;
            if (src1_cont_rows) {
                if (src1_cont_cols) {
                    ggml_fp32_to_fp16_row((float *) src1i, tmp, ne10*ne11);
                }
                else {
                    for (int64_t i01 = 0; i01 < ne11; i01++) {
                        ggml_fp32_to_fp16_row((float *) (src1i + i01*nb11), tmp + i01*ne10, ne10);
                    }
                }
            }
            else {
                for (int64_t i01 = 0; i01 < ne11; i01++) {
                    for (int64_t i00 = 0; i00 < ne10; i00++) {
                        // very slow due to no inlining
                        tmp[i01*ne10 + i00] = ggml_fp32_to_fp16(*(float *) (src1i + i01*nb11 + i00*nb10));
                    }
                }
            }
            ggml_vk_buffer_write(&d_Y, 0, tmp, sizeof(ggml_fp16_t) * y_ne, vk_transfer_queues[1]);

            // Wait for transfers to finish
            vk_transfer_queues[0].queue.waitIdle();
            vk_transfer_queues[1].queue.waitIdle();

            // compute
#ifdef VK_CHK_KERNEL
            auto begin = std::chrono::high_resolution_clock::now();
#endif

            ggml_vk_dispatch_pipeline(vk_pipeline_matmul_f16, {&d_X, &d_Y, &d_D}, { (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01 }, { (uint32_t)ne01, (uint32_t)ne11, 1}, cmd_buffer, fence);
            vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "matmul_f16 waitForFences");

#ifdef VK_CHK_KERNEL
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "m=" << ne01 << " n=" << ne11 << " k=" << ne10 << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms" << std::endl;
#endif

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_transfer_queues[0]);

            vk_device.resetFences({fence});
        }
    }

    vk_device.destroyFence(fence);

    ggml_vk_queue_cleanup(vk_compute_queue);
    ggml_vk_queue_cleanup(vk_transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_transfer_queues[1]);

    if (src0->backend != GGML_BACKEND_GPU) {
        ggml_vk_pool_free(d_X);
    }
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);
}

static void ggml_vk_mul_mat_q_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];
    const ggml_type type = src0->type;
    const bool mul_mat_vec = false;  // ne11 == 1;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;
    const size_t q_sz = ggml_type_size(type) * x_ne / ggml_blck_size(type);

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    if (!mul_mat_vec) {
        ggml_vk_pool_malloc(sizeof(float) * x_ne, &d_X, 0);
    }
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, 0);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, 0);
    vk_buffer d_Q;
    if (src0->backend == GGML_BACKEND_CPU) {
        ggml_vk_pool_malloc(q_sz, &d_Q, 0);
    }

    vk_pipeline* to_fp32_vk = ggml_get_to_fp32_vk(type);
    // vk_pipeline* dmmv = ggml_get_dequantize_mul_mat_vec_vk(type);
    GGML_ASSERT(to_fp32_vk != nullptr);

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_compute_queue);
    vk::Fence fence = vk_device.createFence(vk::FenceCreateFlags{});

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // copy src0 to device if necessary
            if (src0->backend == GGML_BACKEND_CPU) {
                ggml_vk_h2d_tensor_2d(&d_Q, 0, src0, i03, i02, vk_transfer_queues[0]);
            } else if (src0->backend == GGML_BACKEND_GPU) {
                d_Q = *(vk_buffer *) src0->data;
            } else {
                GGML_ASSERT(false);
            }
            if (mul_mat_vec) { // specialized dequantize_mul_mat_vec kernel
                GGML_ASSERT(false);
                // // copy src1 to device
                // events.emplace_back();
                // VK_CHECK(ggml_vk_h2d_tensor_2d(queue, d_Y, 0, src1, i03, i02, events.data() + ev_idx++));

                // // compute
                // const size_t global = ne01 * VK_DMMV_BLOCK_SIZE;
                // const size_t local = VK_DMMV_BLOCK_SIZE;
                // const vk_int ncols = ne00;
                // events.emplace_back();
                // VK_CHECK(vkSetKernelArg(*dmmv, 0, sizeof(vk_buffer), &d_Q));
                // VK_CHECK(vkSetKernelArg(*dmmv, 1, sizeof(float) * local, NULL));
                // VK_CHECK(vkSetKernelArg(*dmmv, 2, sizeof(vk_buffer), &d_Y));
                // VK_CHECK(vkSetKernelArg(*dmmv, 3, sizeof(vk_buffer), &d_D));
                // VK_CHECK(vkSetKernelArg(*dmmv, 4, sizeof(vk_int), &ncols));
                // VK_CHECK(vkEnqueueNDRangeKernel(queue, *dmmv, 1, NULL, &global, &local, events.size() - 1, events.data(), events.data() + ev_idx++));
            } else { // general dequantization kernel + VK matrix matrix multiplication
                // copy src1 to device
                ggml_vk_h2d_tensor_2d(&d_Y, 0, src1, i03, i02, vk_transfer_queues[1]);

                // convert src0 to fp32 on device
                // Wait for transfers to finish
                vk_transfer_queues[0].queue.waitIdle();

                vk_device.resetFences({ fence });
                ggml_vk_dispatch_pipeline(*to_fp32_vk, {&d_Q, &d_X}, { (int)x_ne }, { (uint32_t)x_ne, 1, 1}, cmd_buffer, fence);

                // wait for conversion
                vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "matmul_q_f32 src0 convert waitForFences");

                vk_device.resetFences({ fence });
                cmd_buffer.reset(vk::CommandBufferResetFlags());

                // Wait for transfers to finish
                vk_transfer_queues[1].queue.waitIdle();

                // compute
#ifdef VK_CHK_KERNEL
                auto begin = std::chrono::high_resolution_clock::now();
#endif

                ggml_vk_dispatch_pipeline(vk_pipeline_matmul_f32, {&d_X, &d_Y, &d_D}, { (int)ne01, (int)ne11, (int)ne10, (int)ne00, (int)ne10, (int)ne01 }, { (uint32_t)ne01, (uint32_t)ne11, 1}, cmd_buffer, fence);

                vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "matmul_q_f32 matmul waitForFences");

#ifdef VK_CHK_KERNEL
                auto end = std::chrono::high_resolution_clock::now();

                std::cout << "m=" << ne01 << " n=" << ne11 << " k=" << ne10 << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms" << std::endl;
#endif
            }

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_transfer_queues[0]);
        }
    }

    vk_device.destroyFence(fence);

    ggml_vk_queue_cleanup(vk_compute_queue);
    ggml_vk_queue_cleanup(vk_transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_transfer_queues[1]);

    if (!mul_mat_vec) {
        ggml_vk_pool_free(d_X);
    }
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);
    if (src0->backend == GGML_BACKEND_CPU) {
        ggml_vk_pool_free(d_Q);
    }
}


bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
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
void ggml_vk_test_transfer(size_t ne) {
    // Check transfers are correct
    vk_buffer buffer = ggml_vk_create_buffer(sizeof(float) * ne, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    float* x = (float *) malloc(sizeof(float) * ne);
    float* y = (float *) malloc(sizeof(float) * ne);

    for (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_buffer_write(&buffer, 0, x, sizeof(float) * ne, vk_transfer_queues[0]);

    vk_transfer_queues[0].queue.waitIdle();

    auto end = std::chrono::high_resolution_clock::now();

    double ms_to_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    begin = std::chrono::high_resolution_clock::now();

    ggml_vk_buffer_read(&buffer, 0, y, sizeof(float) * ne, vk_transfer_queues[1]);

    end = std::chrono::high_resolution_clock::now();

    double ms_from_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    double avg_err = 0.0;
    for (size_t i = 0; i < ne; i++) {
        avg_err += std::fabs(x[i] - y[i]);
    }

    double kb = ne * sizeof(float) / 1024.0;

    std::cout << "TEST TRANSFER " << kb << " KB to_gpu " << ms_to_gpu << "ms (" << kb / ms_to_gpu * 1000.0 / 1024.0 << " MB/s) from_gpu " << ms_from_gpu << "ms (" << kb / ms_from_gpu * 1000.0 / 1024.0 << " MB/s) avg_err=" << avg_err / ne << std::endl;

    ggml_vk_destroy_buffer(buffer);

    free(x);
    free(y);
}
void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k) {
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(float) * x_ne, &d_X, 0);
    ggml_vk_pool_malloc(sizeof(float) * y_ne, &d_Y, 0);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, 0);

    float* x = (float *) malloc(sizeof(float) * x_ne);
    float* y = (float *) malloc(sizeof(float) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = rand() / (float)RAND_MAX;
    }

    ggml_vk_buffer_write(&d_X, 0, x, sizeof(float) * x_ne, vk_transfer_queues[0]);
    ggml_vk_buffer_write(&d_Y, 0, y, sizeof(float) * y_ne, vk_transfer_queues[1]);

    // Wait for transfers to finish
    vk_transfer_queues[0].queue.waitIdle();
    vk_transfer_queues[1].queue.waitIdle();

    vk::Fence fence = vk_device.createFence(vk::FenceCreateFlags{});

    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_compute_queue);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_dispatch_pipeline(vk_pipeline_matmul_f32, {&d_X, &d_Y, &d_D}, { (int)m, (int)n, (int)k, (int)k, (int)k, (int)m }, { (uint32_t)m, (uint32_t)n, 1}, cmd_buffer, fence);

    vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "test_matmul_f32 waitForFences");

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_transfer_queues[0]);

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

    ggml_vk_queue_cleanup(vk_compute_queue);
    ggml_vk_queue_cleanup(vk_transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_transfer_queues[1]);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    free(x);
    free(y);
    free(d);
}

void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k) {
    if (!vk_fp16_support) {
        return;
    }
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * x_ne, &d_X, 0);
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * y_ne, &d_Y, 0);
    ggml_vk_pool_malloc(sizeof(float) * d_ne, &d_D, 0);

    ggml_fp16_t* x = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * x_ne);
    ggml_fp16_t* y = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }

    ggml_vk_buffer_write(&d_X, 0, x, sizeof(ggml_fp16_t) * x_ne, vk_transfer_queues[0]);
    ggml_vk_buffer_write(&d_Y, 0, y, sizeof(ggml_fp16_t) * y_ne, vk_transfer_queues[1]);

    vk_transfer_queues[0].queue.waitIdle();
    vk_transfer_queues[1].queue.waitIdle();

    vk::Fence fence = vk_device.createFence(vk::FenceCreateFlags{});
    vk::CommandBuffer cmd_buffer = ggml_vk_cmd_buffer_create(vk_compute_queue);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_dispatch_pipeline(vk_pipeline_matmul_f16, {&d_X, &d_Y, &d_D}, { (int)m, (int)n, (int)k, (int)k, (int)k, (int)m }, { (uint32_t)m, (uint32_t)n, 1}, cmd_buffer, fence);

    vk::resultCheck(vk_device.waitForFences({ fence }, true, uint64_t(-1)), "test_matmul_f16 waitForFences");

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_transfer_queues[0]);

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
            avg_err += std::fabs(d[c * m + r] - d_chk[c * m + r]);
        }
    }

    std::cout << "TEST FP16 m=" << m << " n=" << n << " k=" << k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(fx);
    free(fy);
    free(d_chk);

    vk_device.destroyFence(fence);

    ggml_vk_queue_cleanup(vk_compute_queue);
    ggml_vk_queue_cleanup(vk_transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_transfer_queues[1]);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);

    size_t ev_idx = 0;
    ggml_vk_pool_free(d_D);

    free(x);
    free(y);
    free(d);
}
#endif
