#include "ggml-vulkan.h"

#include <vulkan/vulkan.hpp>
#include "external/vk_mem_alloc.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>

#include "ggml.h"

#define VK_API_VERSION VK_API_VERSION_1_2

vk::Instance vk_instance;
uint32_t vk_compute_queue_family_index;
vk::PhysicalDevice vk_physical_device;
vk::Device vk_device;
vk::DescriptorSetLayout vk_pipeline_matmul_dsl;
vk::Pipeline vk_pipeline_matmul;
VmaAllocation vk_buffer_qa_alloc, vk_buffer_a_alloc, vk_buffer_b_alloc, vk_buffer_c_alloc;
vk::Buffer vk_buffer_qa, vk_buffer_a, vk_buffer_b, vk_buffer_c;

void ggml_vk_init(void) {
    char* GGML_VULKAN_DEVICE = getenv("GGML_VULKAN_DEVICE");
    int dev_num = (GGML_VULKAN_DEVICE == NULL ? 0 : atoi(GGML_VULKAN_DEVICE));

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION };
    const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers.size(), layers.data());
    vk_instance = vk::createInstance(instance_create_info);

    vk_physical_device = vk_instance.enumeratePhysicalDevices()[dev_num];
    vk::PhysicalDeviceProperties device_props = vk_physical_device.getProperties();
    std::cout << "ggml_vulkan: Using " << device_props.deviceName << std::endl;

    std::vector<vk::QueueFamilyProperties> queue_family_props = vk_physical_device.getQueueFamilyProperties();
    auto prop_it = std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const vk::QueueFamilyProperties& prop)
    {
        return prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    vk_compute_queue_family_index = std::distance(queue_family_props.begin(), prop_it);

    const float queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo device_queue_create_info(vk::DeviceQueueCreateFlags(), vk_compute_queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), device_queue_create_info);
    vk_device = vk_physical_device.createDevice(device_create_info);

    std::vector<char> matmul_shader_contents;
    if (std::ifstream shader_file{ "ggml-vulkan-matmul.spv", std::ios::binary | std::ios::ate }) {
        const size_t file_size = shader_file.tellg();
        shader_file.seekg(0);
        matmul_shader_contents.resize(file_size, '\0');
        shader_file.read(matmul_shader_contents.data(), file_size);
    }

    vk::ShaderModuleCreateInfo shader_module_create_info(
        vk::ShaderModuleCreateFlags(),
        matmul_shader_contents.size(),
        reinterpret_cast<const uint32_t*>(matmul_shader_contents.data())
    );
    vk::ShaderModule shader_module = vk_device.createShaderModule(shader_module_create_info);

    const std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_binding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
        vk::DescriptorSetLayoutCreateFlags(),
        descriptor_set_layout_binding);
    vk_pipeline_matmul_dsl = vk_device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), vk_pipeline_matmul_dsl);
    vk::PipelineLayout pipeline_layout = vk_device.createPipelineLayout(pipeline_layout_create_info);
    vk::PipelineCache pipeline_cache = vk_device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shader_module,
            "main");
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),    // Flags
        pipeline_shader_create_info,     // Shader Create Info struct
        pipeline_layout);              // Pipeline Layout
    vk_pipeline_matmul = vk_device.createComputePipeline(pipeline_cache, compute_pipeline_create_info).value;
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

struct vk_buffer {
    vk::Buffer buffer;
    size_t size = 0;
};

static vk_buffer g_vk_buffer_pool[MAX_VK_BUFFERS];
static std::atomic_flag g_vk_pool_lock = ATOMIC_FLAG_INIT;

static vk::Buffer ggml_vk_pool_malloc(size_t size, size_t * actual_size) {
    scoped_spin_lock lock(g_vk_pool_lock);

    int best_i = -1;
    size_t best_size = std::numeric_limits<size_t>::max(); //smallest unused buffer that fits our needs
    int worst_i = -1;
    size_t worst_size = 0; //largest unused buffer seen so far
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer &b = g_vk_buffer_pool[i];
        if (b.size > 0 && b.size >= size && b.size < best_size)
        {
            best_i = i;
            best_size = b.size;
        }
        if (b.size > 0 && b.size > worst_size)
        {
            worst_i = i;
            worst_size = b.size;
        }
    }
    if(best_i!=-1) //found the smallest buffer that fits our needs
    {
        vk_buffer& b = g_vk_buffer_pool[best_i];
        vk::Buffer buffer = b.buffer;
        *actual_size = b.size;
        b.size = 0;
        return buffer;
    }
    if(worst_i!=-1) //no buffer that fits our needs, resize largest one to save memory
    {
         vk_buffer& b = g_vk_buffer_pool[worst_i];
         vk::Buffer buffer = b.buffer;
         b.size = 0;
         // vkReleaseMemObject(buffer);
    }
    vk::Buffer buffer;

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::SharingMode::eExclusive,
        1,
        &vk_compute_queue_family_index
    };

    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.vulkanApiVersion = VK_API_VERSION;
    allocator_info.physicalDevice = vk_physical_device;
    allocator_info.device = vk_device;
    allocator_info.instance = vk_instance;

    VmaAllocator allocator;
    vmaCreateAllocator(&allocator_info, &allocator);

    VmaAllocationCreateInfo allocation_info = {};
    allocation_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VmaAllocation buffer_allocation;
    vmaCreateBuffer(allocator,
                    &static_cast<VkBufferCreateInfo>(buffer_create_info),
                    &allocation_info,
                    &static_cast<VkBuffer>(buffer),
                    &buffer_allocation,
                    nullptr);

    *actual_size = size;
    return buffer;
}

static void ggml_vk_pool_free(vk::Buffer buffer, size_t size) {
    scoped_spin_lock lock(g_vk_pool_lock);

    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer& b = g_vk_buffer_pool[i];
        if (b.size == 0) {
            b.buffer = buffer;
            b.size = size;
            return;
        }
    }
    fprintf(stderr, "WARNING: vk buffer pool full, increase MAX_VK_BUFFERS\n");
    vkReleaseMemObject(mem);
}
