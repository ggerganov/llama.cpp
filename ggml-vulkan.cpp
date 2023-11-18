#include "ggml-vulkan.h"

#ifdef VK_CHK_KERNEL
#include <cblas.h>
#include <chrono>
#endif

#ifdef VK_PROFILE
#define PROFILE(name, block) do { \
    auto begin = std::chrono::steady_clock::now(); \
    block \
    auto end = std::chrono::steady_clock::now(); \
    double time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0; \
    printf("%s: %lf ms\n", name, time_taken); \
} while(0)
#else
#define PROFILE(name, block) block
#endif

#include <vulkan/vulkan.hpp>

#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <vector>
#include <sstream>
#include <utility>

#include "ggml.h"

#include "ggml-vulkan-shaders.hpp"

#define VK_API_VERSION VK_API_VERSION_1_2

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define VK_TRANSFER_QUEUE_COUNT 2

#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN 0
#define VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI 1
#define VK_DEVICE_DESCRIPTOR_POOL_MODE_SINGLE 2

#define VK_NUM_TYPES 16

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 1
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

typedef void (*ggml_vk_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

struct vk_buffer {
    vk::Buffer buffer;
    vk::DeviceMemory device_memory;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;
    // Staging buffers
    vk_buffer * sb_write;
    vk_buffer * sb_read;
    uint32_t qf_owner;
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint32_t offset;
    uint32_t size;
};

struct vk_pipeline {
    std::string name;
    vk::DescriptorSetLayout dsl;
    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::DescriptorSet> descriptor_sets;
    uint32_t descriptor_set_idx;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
};

struct vk_queue {
    uint32_t queue_family_index;
    vk::Queue queue;
    vk::CommandPool pool;
    uint32_t cmd_buffer_idx;
    std::vector<vk::CommandBuffer> cmd_buffers;

    vk::PipelineStageFlags stage_flags;
};

struct vk_semaphore {
    vk::Semaphore s;
    uint64_t value;
};

struct vk_submission {
    vk::CommandBuffer buffer;
    std::vector<vk_semaphore> wait_semaphores;
    std::vector<vk_semaphore> signal_semaphores;
};

typedef std::vector<vk_submission> vk_sequence;

struct vk_device {
    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    bool fp16;
    vk::Device device;
    uint32_t vendor_id;
    vk_queue compute_queue;
    vk_queue transfer_queues[VK_TRANSFER_QUEUE_COUNT];
    uint32_t descriptor_set_mode;
};

struct vk_op_push_constants {
    int M;
    int N;
    float param;
};

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct ggml_vk_tensor_extra_gpu {
    std::vector<vk_staging_memcpy> memcpys;
    std::vector<vk_sequence> in0_seqs;
    std::vector<vk_sequence> in1_seqs;
    std::vector<vk_sequence> comp_seqs;
    std::vector<vk_sequence> out_seqs;

    size_t tensor_size;
    vk_buffer * gpu_buffer;
};

struct ggml_vk_garbage_collector {
    std::vector<vk_pipeline *> pipelines;
    std::vector<ggml_vk_tensor_extra_gpu *> extras;
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
};

vk::Instance vk_instance;
vk_device vk_device;
vk_pipeline vk_pipeline_matmul_f32_l, vk_pipeline_matmul_f32_m, vk_pipeline_matmul_f32_s;
vk_pipeline vk_pipeline_matmul_f32_aligned_l, vk_pipeline_matmul_f32_aligned_m, vk_pipeline_matmul_f32_aligned_s;
vk_pipeline vk_pipeline_matmul_f16_l, vk_pipeline_matmul_f16_m, vk_pipeline_matmul_f16_s;
vk_pipeline vk_pipeline_matmul_f16_aligned_l, vk_pipeline_matmul_f16_aligned_m, vk_pipeline_matmul_f16_aligned_s;
vk_pipeline vk_pipeline_matmul_f16_f32_l, vk_pipeline_matmul_f16_f32_m, vk_pipeline_matmul_f16_f32_s;
vk_pipeline vk_pipeline_matmul_f16_f32_aligned_l, vk_pipeline_matmul_f16_f32_aligned_m, vk_pipeline_matmul_f16_f32_aligned_s;
vk_pipeline vk_pipeline_matmul_split_k_reduce;
vk_pipeline vk_pipeline_dequant[VK_NUM_TYPES];
vk_pipeline vk_pipeline_dequant_mul_mat_vec[VK_NUM_TYPES];
vk_pipeline vk_pipeline_dequant_mul_mat_vec_f32[VK_NUM_TYPES];
vk_pipeline vk_pipeline_get_rows[VK_NUM_TYPES];
vk_pipeline vk_pipeline_get_rows_f32[VK_NUM_TYPES];
vk_pipeline vk_pipeline_mul_f32;
vk_pipeline vk_pipeline_add_f32, vk_pipeline_add_f16_f32_f16;
vk_pipeline vk_pipeline_scale_f32;
vk_pipeline vk_pipeline_rms_norm_f32;

static size_t vk_semaphore_idx;
static ggml_vk_garbage_collector vk_gc;
static std::vector<std::tuple<void*, size_t, vk_buffer>> vk_pinned_memory;
static size_t vk_prealloc_size_d, vk_prealloc_size_qx, vk_prealloc_size_qy, vk_prealloc_size_x, vk_prealloc_size_y;
static vk_buffer vk_prealloc_d, vk_prealloc_qx, vk_prealloc_qy, vk_prealloc_x, vk_prealloc_y;
static vk::Fence vk_fence;

static vk_pipeline ggml_vk_create_pipeline(const std::string& name, size_t spv_size, const void* spv_data, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, std::vector<int>&& specialization_constants, uint32_t align) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_pipeline(" << name << ", " << entrypoint << ", " << parameter_count << ", " << push_constant_size << ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " << align << ")" << std::endl;
#endif
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk_pipeline pipeline;

    pipeline.name = name;
    pipeline.parameter_count = parameter_count;
    pipeline.push_constant_size = push_constant_size;
    pipeline.wg_denoms = wg_denoms;
    pipeline.align = align;

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));
    vk::ShaderModule shader_module = vk_device.device.createShaderModule(shader_module_create_info);

    std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
    std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
    for (uint32_t i = 0; i < parameter_count; i++) {
        dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        dsl_binding_flags.push_back({});
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline.push_constant_size
    );

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
        {},
        dsl_binding);
    descriptor_set_layout_create_info.setPNext(&dslbfci);
    pipeline.dsl = vk_device.device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    // Check if device supports multiple descriptors per pool
    if (vk_device.descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN) {
        const uint32_t alloc_count = 2;

        // Try allocating multiple sets from one pool
        // This fails on AMD for some reason, so add a fall back to allocating one pool per set
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline.parameter_count);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, alloc_count, descriptor_pool_size);
        vk::DescriptorPool pool = vk_device.device.createDescriptorPool(descriptor_pool_create_info);

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = pipeline.dsl;
        }
        try {
            vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pool, alloc_count, layouts.data());
            std::vector<vk::DescriptorSet> sets = vk_device.device.allocateDescriptorSets(descriptor_set_alloc_info);
        } catch(vk::OutOfPoolMemoryError const&) {
            vk_device.descriptor_set_mode = VK_DEVICE_DESCRIPTOR_POOL_MODE_SINGLE;
        }

        vk_device.device.destroyDescriptorPool(pool);
    }

    if (vk_device.descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI) {
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline.parameter_count);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, 128, descriptor_pool_size);
        pipeline.descriptor_pools.push_back(vk_device.device.createDescriptorPool(descriptor_pool_create_info));
    }

    pipeline.descriptor_set_idx = 0;

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), pipeline.dsl, pcr);
    pipeline.layout = vk_device.device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(int);
        specialization_entries[i].size = sizeof(int);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(int),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shader_module,
            entrypoint.c_str(),
            &specialization_info);
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),
        pipeline_shader_create_info,
        pipeline.layout);
    pipeline.pipeline = vk_device.device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;

    return pipeline;
}

static void ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline& pipeline, uint32_t n) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_pipeline_allocate_descriptor_sets(" << pipeline.name << ", " << n << ")" << std::endl;
#endif
    // Check if gc already contains pipeline before adding it
    bool gc_found = false;
    for (auto * pl : vk_gc.pipelines) {
        if (&pipeline == pl) {
            gc_found = true;
            break;
        }
    }

    if (!gc_found) {
        vk_gc.pipelines.push_back(&pipeline);
    }

    if (pipeline.descriptor_sets.size() >= pipeline.descriptor_set_idx + n) {
        // Enough descriptors are available
        return;
    }

    if (vk_device.descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI) {
        const uint32_t alloc_count = pipeline.descriptor_set_idx + n - pipeline.descriptor_sets.size();

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = pipeline.dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pipeline.descriptor_pools[0], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = vk_device.device.allocateDescriptorSets(descriptor_set_alloc_info);
        pipeline.descriptor_sets.insert(pipeline.descriptor_sets.end(), sets.begin(), sets.end());
    } else {
        for (uint32_t i = pipeline.descriptor_sets.size(); i < pipeline.descriptor_set_idx + n; i++) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline.parameter_count);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, 1, descriptor_pool_size);
            pipeline.descriptor_pools.push_back(vk_device.device.createDescriptorPool(descriptor_pool_create_info));

            vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pipeline.descriptor_pools[i], 1, &pipeline.dsl);
            std::vector<vk::DescriptorSet> sets = vk_device.device.allocateDescriptorSets(descriptor_set_alloc_info);
            pipeline.descriptor_sets.push_back(sets[0]);
        }
    }
}

static void ggml_vk_pipeline_cleanup(vk_pipeline& pipeline) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_pipeline_cleanup(" << pipeline.name << ")" << std::endl;
#endif
    pipeline.descriptor_set_idx = 0;
}

static vk::CommandBuffer ggml_vk_create_cmd_buffer(vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_cmd_buffer()" << std::endl;
#endif
    if (q.cmd_buffers.size() > q.cmd_buffer_idx) {
        // Reuse command buffer
        return q.cmd_buffers[q.cmd_buffer_idx++];
    }

    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        q.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = vk_device.device.allocateCommandBuffers(command_buffer_alloc_info);
    auto buf = cmd_buffers.front();

    q.cmd_buffers.push_back(buf);
    q.cmd_buffer_idx++;

    return buf;
}

static vk_submission ggml_vk_create_submission(vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_submission()" << std::endl;
#endif
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(q);
    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
    return s;
}

static vk_sequence ggml_vk_create_sequence_1(vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_sequence_1()" << std::endl;
#endif
    return { ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores)) };
}

static void ggml_vk_submit(vk_queue& q, std::vector<vk_sequence>& sequences, vk::Fence fence) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_submit(" << q.queue_family_index << ", (" << q.queue << "), " << sequences.size() << ")" << std::endl;
#endif
    if (sequences.empty()) {
        return;
    }

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    for (const auto& sequence : sequences) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    for (const auto& sequence : sequences) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(q.stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
                &tl_submit_infos[idx],
            });
        }
    }

    q.queue.submit(submit_infos, fence);

    sequences.clear();
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_find_queue_family_index()" << std::endl;
#endif
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to ignoring min_num_queries
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static vk_queue ggml_vk_create_queue(uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_queue()" << std::endl;
#endif
    vk_queue q;
    q.queue_family_index = queue_family_index;

    vk::CommandPoolCreateInfo command_pool_create_info_compute(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT), queue_family_index);
    q.pool = vk_device.device.createCommandPool(command_pool_create_info_compute);

    q.cmd_buffer_idx = 0;

    q.queue = vk_device.device.getQueue(queue_family_index, queue_index);

    q.stage_flags = stage_flags;

    return q;
}

static vk_semaphore * ggml_vk_create_binary_semaphore() {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_timeline_semaphore()" << std::endl;
#endif
    vk::SemaphoreTypeCreateInfo info{ vk::SemaphoreType::eBinary, 0 };
    vk::Semaphore semaphore = vk_device.device.createSemaphore(vk::SemaphoreCreateInfo{ {}, &info });
    vk_gc.semaphores.push_back({ semaphore, 0 });
    return &vk_gc.semaphores[vk_gc.semaphores.size() - 1];
}

static vk_semaphore * ggml_vk_create_timeline_semaphore() {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_timeline_semaphore()" << std::endl;
#endif
    if (vk_semaphore_idx >= vk_gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo info{ vk::SemaphoreType::eTimeline, 0 };
        vk::Semaphore semaphore = vk_device.device.createSemaphore(vk::SemaphoreCreateInfo{ {}, &info });
        vk_gc.tl_semaphores.push_back({ semaphore, 0 });
        return &vk_gc.tl_semaphores[vk_semaphore_idx++];
    }
    return &vk_gc.tl_semaphores[vk_semaphore_idx++];
}

static void ggml_vk_queue_cleanup(vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_queue_cleanup()" << std::endl;
#endif
    // Requires command buffers to be done

    vk_device.device.resetCommandPool(q.pool);
    q.cmd_buffer_idx = 0;
}

static vk_buffer ggml_vk_create_buffer(size_t size, vk::MemoryPropertyFlags req_flags) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_buffer(" << size << ", " << to_string(req_flags) << ")" << std::endl;
#endif
    vk_buffer buf;

    buf.size = size;
    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    buf.buffer = vk_device.device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = vk_device.device.getBufferMemoryRequirements(buf.buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = vk_device.physical_device.getMemoryProperties();

    uint32_t memory_type_index = uint32_t(~0);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props.memoryTypes[i];
        if ((mem_req.memoryTypeBits & ((uint64_t)1 << i)) && (req_flags & memory_type.propertyFlags) == req_flags && mem_props.memoryHeaps[memory_type.heapIndex].size >= mem_req.size) {
            memory_type_index = i;
            break;
        }
    }

    buf.device_memory = vk_device.device.allocateMemory({ mem_req.size, memory_type_index });
    buf.memory_property_flags = req_flags;
    buf.ptr = nullptr;

    if (req_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        buf.ptr = vk_device.device.mapMemory(buf.device_memory, 0, VK_WHOLE_SIZE);
    }

    vk_device.device.bindBufferMemory(buf.buffer, buf.device_memory, 0);

    buf.sb_write = nullptr;
    buf.sb_read = nullptr;

    buf.qf_owner = VK_QUEUE_FAMILY_IGNORED;

    return buf;
}

static vk_subbuffer ggml_vk_subbuffer(vk_buffer& buf) {
    return { buf, 0, (uint32_t) buf.size };
}

static void ggml_vk_sync_buffers(vk::CommandBuffer& cmd_buffer, std::vector<vk_subbuffer>&& buffers, vk_queue& q, vk::AccessFlags&& src_mask, vk::AccessFlags&& dst_mask, bool force_sync) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_sync_buffers()" << std::endl;
#endif
    std::vector<vk::BufferMemoryBarrier> bmem_barriers;

    uint32_t sfi;
    uint32_t dfi;

    for (auto& buf : buffers) {
        if (buf.buffer.qf_owner != VK_QUEUE_FAMILY_IGNORED && buf.buffer.qf_owner != q.queue_family_index) {
            sfi = buf.buffer.qf_owner;
            dfi = q.queue_family_index;
            buf.buffer.qf_owner = dfi;
            bmem_barriers.push_back({ src_mask, dst_mask, sfi, dfi, buf.buffer.buffer, buf.offset, buf.size });
        } else if (force_sync) {
            sfi = VK_QUEUE_FAMILY_IGNORED;
            dfi = VK_QUEUE_FAMILY_IGNORED;
            bmem_barriers.push_back({ src_mask, dst_mask, sfi, dfi, buf.buffer.buffer, buf.offset, buf.size });
        }
    }

    if (bmem_barriers.empty()) {
        return;
    }

    cmd_buffer.pipelineBarrier(
        q.stage_flags,
        q.stage_flags,
        {},
        {},
        bmem_barriers,
        {}
    );
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_destroy_buffer(" << buf.size << ")" << std::endl;
#endif
    buf.size = 0;
    vk_device.device.freeMemory(buf.device_memory);
    vk_device.device.destroyBuffer(buf.buffer);

    // Cleanup staging buffers
    if (buf.sb_write != nullptr) {
        vk_device.device.freeMemory(buf.sb_write->device_memory);
        vk_device.device.destroyBuffer(buf.sb_write->buffer);
        delete buf.sb_write;
        buf.sb_write = nullptr;
    }
    if (buf.sb_read != nullptr) {
        vk_device.device.freeMemory(buf.sb_read->device_memory);
        vk_device.device.destroyBuffer(buf.sb_read->buffer);
        delete buf.sb_read;
        buf.sb_read = nullptr;
    }
}

static bool ggml_vk_build_shader(ggml_type type) {
    switch(type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
        return true;
    default:
        return false;
    }
}

static void ggml_vk_load_shaders() {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_load_shaders()" << std::endl;
#endif

    // mulmat
    auto warptile_l = { 128, 128, 128, 16, 64, 64, 2, 4, 4 };
    auto warptile_m = { 128,  64,  64, 16, 32, 32, 2, 4, 2 };
    auto warptile_s = {  32,  32,  32,  8, 32, 32, 2, 2, 2 };

    if (vk_device.fp16) {
        vk_pipeline_matmul_f32_l = ggml_vk_create_pipeline("matmul_f32_l", matmul_f32_l_len, matmul_f32_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f32_m = ggml_vk_create_pipeline("matmul_f32_m", matmul_f32_m_len, matmul_f32_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f32_s = ggml_vk_create_pipeline("matmul_f32_s", matmul_f32_s_len, matmul_f32_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);
        vk_pipeline_matmul_f32_aligned_l = ggml_vk_create_pipeline("matmul_f32_aligned_l", matmul_f32_aligned_l_len, matmul_f32_aligned_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f32_aligned_m = ggml_vk_create_pipeline("matmul_f32_aligned_m", matmul_f32_aligned_m_len, matmul_f32_aligned_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f32_aligned_s = ggml_vk_create_pipeline("matmul_f32_aligned_s", matmul_f32_aligned_s_len, matmul_f32_aligned_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_f16_l = ggml_vk_create_pipeline("matmul_f16_l", matmul_f16_l_len, matmul_f16_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f16_m = ggml_vk_create_pipeline("matmul_f16_m", matmul_f16_m_len, matmul_f16_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f16_s = ggml_vk_create_pipeline("matmul_f16_s", matmul_f16_s_len, matmul_f16_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);

        vk_pipeline_matmul_f16_aligned_l = ggml_vk_create_pipeline("matmul_f16_aligned_l", matmul_f16_aligned_l_len, matmul_f16_aligned_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f16_aligned_m = ggml_vk_create_pipeline("matmul_f16_aligned_m", matmul_f16_aligned_m_len, matmul_f16_aligned_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f16_aligned_s = ggml_vk_create_pipeline("matmul_f16_aligned_s", matmul_f16_aligned_s_len, matmul_f16_aligned_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_f16_f32_l = ggml_vk_create_pipeline("matmul_f16_f32_l", matmul_f16_f32_l_len, matmul_f16_f32_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f16_f32_m = ggml_vk_create_pipeline("matmul_f16_f32_m", matmul_f16_f32_m_len, matmul_f16_f32_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f16_f32_s = ggml_vk_create_pipeline("matmul_f16_f32_s", matmul_f16_f32_s_len, matmul_f16_f32_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);
        vk_pipeline_matmul_f16_f32_aligned_l = ggml_vk_create_pipeline("matmul_f16_f32_aligned_l", matmul_f16_f32_aligned_l_len, matmul_f16_f32_aligned_l_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f16_f32_aligned_m = ggml_vk_create_pipeline("matmul_f16_f32_aligned_m", matmul_f16_f32_aligned_m_len, matmul_f16_f32_aligned_m_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f16_f32_aligned_s = ggml_vk_create_pipeline("matmul_f16_f32_aligned_s", matmul_f16_f32_aligned_s_len, matmul_f16_f32_aligned_s_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_split_k_reduce = ggml_vk_create_pipeline("split_k_reduce", split_k_reduce_fp32_len, split_k_reduce_fp32_data, "main", 1, 3 * sizeof(int), {32, 32, 1}, {}, 1);

        // Build dequant shaders
        vk_pipeline_dequant[GGML_TYPE_F32] = ggml_vk_create_pipeline("f32_to_f16", f32_to_f16_len, f32_to_f16_data, "main", 2, 4 * sizeof(int), {64, 1, 1}, {}, 1);

        vk_pipeline_dequant[GGML_TYPE_F16] = ggml_vk_create_pipeline("dequant_f16", dequant_f16_len, dequant_f16_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q2_K] = ggml_vk_create_pipeline("dequant_q2_K", dequant_q2_K_len, dequant_q2_K_data, "main", 2, 4 * sizeof(int), {256 * 64, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q3_K] = ggml_vk_create_pipeline("dequant_q3_K", dequant_q3_K_len, dequant_q3_K_data, "main", 2, 4 * sizeof(int), {256 * 64, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_K] = ggml_vk_create_pipeline("dequant_q4_K", dequant_q4_K_len, dequant_q4_K_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_K] = ggml_vk_create_pipeline("dequant_q5_K", dequant_q5_K_len, dequant_q5_K_data, "main", 2, 4 * sizeof(int), {256 * 64, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q6_K] = ggml_vk_create_pipeline("dequant_q6_K", dequant_q6_K_len, dequant_q6_K_data, "main", 2, 4 * sizeof(int), {256 * 64, 1, 1}, {}, 1);

        // get_rows
        vk_pipeline_get_rows[GGML_TYPE_F16] = ggml_vk_create_pipeline("get_rows_f16", get_rows_f16_len, get_rows_f16_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        vk_pipeline_get_rows_f32[GGML_TYPE_F16] = ggml_vk_create_pipeline("get_rows_f16_f32", get_rows_f16_f32_len, get_rows_f16_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        // add
        vk_pipeline_add_f32 = ggml_vk_create_pipeline("add_f32", add_f32_len, add_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_add_f16_f32_f16 = ggml_vk_create_pipeline("add_f16_f32_f16", add_f16_f32_f16_len, add_f16_f32_f16_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        // Static shaders
        vk_pipeline_mul_f32 = ggml_vk_create_pipeline("mul_f32", mul_f32_len, mul_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        vk_pipeline_scale_f32 = ggml_vk_create_pipeline("scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    } else {
        vk_pipeline_matmul_f32_l = ggml_vk_create_pipeline("matmul_f32_l", matmul_f32_l_fp32_len, matmul_f32_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f32_m = ggml_vk_create_pipeline("matmul_f32_m", matmul_f32_m_fp32_len, matmul_f32_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f32_s = ggml_vk_create_pipeline("matmul_f32_s", matmul_f32_s_fp32_len, matmul_f32_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);
        vk_pipeline_matmul_f32_aligned_l = ggml_vk_create_pipeline("matmul_f32_aligned_l", matmul_f32_aligned_l_fp32_len, matmul_f32_aligned_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f32_aligned_m = ggml_vk_create_pipeline("matmul_f32_aligned_m", matmul_f32_aligned_m_fp32_len, matmul_f32_aligned_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f32_aligned_s = ggml_vk_create_pipeline("matmul_f32_aligned_s", matmul_f32_aligned_s_fp32_len, matmul_f32_aligned_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_f16_l = ggml_vk_create_pipeline("matmul_f16_l", matmul_f16_l_fp32_len, matmul_f16_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f16_m = ggml_vk_create_pipeline("matmul_f16_m", matmul_f16_m_fp32_len, matmul_f16_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f16_s = ggml_vk_create_pipeline("matmul_f16_s", matmul_f16_s_fp32_len, matmul_f16_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);

        vk_pipeline_matmul_f16_aligned_l = ggml_vk_create_pipeline("matmul_f16_aligned_l", matmul_f16_aligned_l_fp32_len, matmul_f16_aligned_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f16_aligned_m = ggml_vk_create_pipeline("matmul_f16_aligned_m", matmul_f16_aligned_m_fp32_len, matmul_f16_aligned_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f16_aligned_s = ggml_vk_create_pipeline("matmul_f16_aligned_s", matmul_f16_aligned_s_fp32_len, matmul_f16_aligned_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_f16_f32_l = ggml_vk_create_pipeline("matmul_f16_f32_l", matmul_f16_f32_l_fp32_len, matmul_f16_f32_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 1);
        vk_pipeline_matmul_f16_f32_m = ggml_vk_create_pipeline("matmul_f16_f32_m", matmul_f16_f32_m_fp32_len, matmul_f16_f32_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 1);
        vk_pipeline_matmul_f16_f32_s = ggml_vk_create_pipeline("matmul_f16_f32_s", matmul_f16_f32_s_fp32_len, matmul_f16_f32_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 1);
        vk_pipeline_matmul_f16_f32_aligned_l = ggml_vk_create_pipeline("matmul_f16_f32_aligned_l", matmul_f16_f32_aligned_l_fp32_len, matmul_f16_f32_aligned_l_fp32_data, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
        vk_pipeline_matmul_f16_f32_aligned_m = ggml_vk_create_pipeline("matmul_f16_f32_aligned_m", matmul_f16_f32_aligned_m_fp32_len, matmul_f16_f32_aligned_m_fp32_data, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
        vk_pipeline_matmul_f16_f32_aligned_s = ggml_vk_create_pipeline("matmul_f16_f32_aligned_s", matmul_f16_f32_aligned_s_fp32_len, matmul_f16_f32_aligned_s_fp32_data, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

        vk_pipeline_matmul_split_k_reduce = ggml_vk_create_pipeline("split_k_reduce", split_k_reduce_fp32_len, split_k_reduce_fp32_data, "main", 1, 3 * sizeof(int), {32, 32, 1}, {}, 1);

        // Build dequant shaders
        vk_pipeline_dequant[GGML_TYPE_F32] = ggml_vk_create_pipeline("f32_to_f16", f32_to_f16_fp32_len, f32_to_f16_fp32_data, "main", 2, 4 * sizeof(int), {64, 1, 1}, {}, 1);

        vk_pipeline_dequant[GGML_TYPE_F16] = ggml_vk_create_pipeline("dequant_f16", dequant_f16_fp32_len, dequant_f16_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("dequant_q4_0", dequant_q4_0_fp32_len, dequant_q4_0_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("dequant_q4_1", dequant_q4_1_fp32_len, dequant_q4_1_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("dequant_q5_0", dequant_q5_0_fp32_len, dequant_q5_0_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("dequant_q5_1", dequant_q5_1_fp32_len, dequant_q5_1_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("dequant_q8_0", dequant_q8_0_fp32_len, dequant_q8_0_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q2_K] = ggml_vk_create_pipeline("dequant_q2_K", dequant_q2_K_fp32_len, dequant_q2_K_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q3_K] = ggml_vk_create_pipeline("dequant_q3_K", dequant_q3_K_fp32_len, dequant_q3_K_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q4_K] = ggml_vk_create_pipeline("dequant_q4_K", dequant_q4_K_fp32_len, dequant_q4_K_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q5_K] = ggml_vk_create_pipeline("dequant_q5_K", dequant_q5_K_fp32_len, dequant_q5_K_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
        vk_pipeline_dequant[GGML_TYPE_Q6_K] = ggml_vk_create_pipeline("dequant_q6_K", dequant_q6_K_fp32_len, dequant_q6_K_fp32_data, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);

        // get_rows
        vk_pipeline_get_rows[GGML_TYPE_F16] = ggml_vk_create_pipeline("get_rows_f16", get_rows_f16_fp32_len, get_rows_f16_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("get_rows_q4_0", get_rows_q4_0_fp32_len, get_rows_q4_0_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("get_rows_q4_1", get_rows_q4_1_fp32_len, get_rows_q4_1_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("get_rows_q5_0", get_rows_q5_0_fp32_len, get_rows_q5_0_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("get_rows_q5_1", get_rows_q5_1_fp32_len, get_rows_q5_1_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("get_rows_q8_0", get_rows_q8_0_fp32_len, get_rows_q8_0_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        vk_pipeline_get_rows_f32[GGML_TYPE_F16] = ggml_vk_create_pipeline("get_rows_f16_f32", get_rows_f16_f32_fp32_len, get_rows_f16_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("get_rows_q4_0_f32", get_rows_q4_0_f32_fp32_len, get_rows_q4_0_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("get_rows_q4_1_f32", get_rows_q4_1_f32_fp32_len, get_rows_q4_1_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("get_rows_q5_0_f32", get_rows_q5_0_f32_fp32_len, get_rows_q5_0_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("get_rows_q5_1_f32", get_rows_q5_1_f32_fp32_len, get_rows_q5_1_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_get_rows_f32[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("get_rows_q8_0_f32", get_rows_q8_0_f32_fp32_len, get_rows_q8_0_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        // add
        vk_pipeline_add_f32 = ggml_vk_create_pipeline("add_f32", add_f32_fp32_len, add_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
        vk_pipeline_add_f16_f32_f16 = ggml_vk_create_pipeline("add_f16_f32_f16", add_f16_f32_f16_fp32_len, add_f16_f32_f16_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        // Static shaders
        vk_pipeline_mul_f32 = ggml_vk_create_pipeline("mul_f32", mul_f32_fp32_len, mul_f32_fp32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

        vk_pipeline_scale_f32 = ggml_vk_create_pipeline("scale_f32", scale_f32_fp32_len, scale_f32_fp32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    }

    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_F16] = ggml_vk_create_pipeline("mul_mat_vec_f16", mul_mat_vec_f16_len, mul_mat_vec_f16_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("mul_mat_vec_q4_0", mul_mat_vec_q4_0_len, mul_mat_vec_q4_0_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("mul_mat_vec_q4_1", mul_mat_vec_q4_1_len, mul_mat_vec_q4_1_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("mul_mat_vec_q5_0", mul_mat_vec_q5_0_len, mul_mat_vec_q5_0_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("mul_mat_vec_q5_1", mul_mat_vec_q5_1_len, mul_mat_vec_q5_1_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("mul_mat_vec_q8_0", mul_mat_vec_q8_0_len, mul_mat_vec_q8_0_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q2_K] = ggml_vk_create_pipeline("mul_mat_vec_q2_K", mul_mat_vec_q2_K_len, mul_mat_vec_q2_K_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q3_K] = ggml_vk_create_pipeline("mul_mat_vec_q3_K", mul_mat_vec_q3_K_len, mul_mat_vec_q3_K_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q4_K] = ggml_vk_create_pipeline("mul_mat_vec_q4_K", mul_mat_vec_q4_K_len, mul_mat_vec_q4_K_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q5_K] = ggml_vk_create_pipeline("mul_mat_vec_q5_K", mul_mat_vec_q5_K_len, mul_mat_vec_q5_K_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec[GGML_TYPE_Q6_K] = ggml_vk_create_pipeline("mul_mat_vec_q6_K", mul_mat_vec_q6_K_len, mul_mat_vec_q6_K_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);

    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_F16] = ggml_vk_create_pipeline("mul_mat_vec_f16_f32", mul_mat_vec_f16_f32_len, mul_mat_vec_f16_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q4_0] = ggml_vk_create_pipeline("mul_mat_vec_q4_0_f32", mul_mat_vec_q4_0_f32_len, mul_mat_vec_q4_0_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q4_1] = ggml_vk_create_pipeline("mul_mat_vec_q4_1_f32", mul_mat_vec_q4_1_f32_len, mul_mat_vec_q4_1_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q5_0] = ggml_vk_create_pipeline("mul_mat_vec_q5_0_f32", mul_mat_vec_q5_0_f32_len, mul_mat_vec_q5_0_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q5_1] = ggml_vk_create_pipeline("mul_mat_vec_q5_1_f32", mul_mat_vec_q5_1_f32_len, mul_mat_vec_q5_1_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q8_0] = ggml_vk_create_pipeline("mul_mat_vec_q8_0_f32", mul_mat_vec_q8_0_f32_len, mul_mat_vec_q8_0_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q2_K] = ggml_vk_create_pipeline("mul_mat_vec_q2_K_f32", mul_mat_vec_q2_K_f32_len, mul_mat_vec_q2_K_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q3_K] = ggml_vk_create_pipeline("mul_mat_vec_q3_K_f32", mul_mat_vec_q3_K_f32_len, mul_mat_vec_q3_K_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q4_K] = ggml_vk_create_pipeline("mul_mat_vec_q4_K_f32", mul_mat_vec_q4_K_f32_len, mul_mat_vec_q4_K_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q5_K] = ggml_vk_create_pipeline("mul_mat_vec_q5_K_f32", mul_mat_vec_q5_K_f32_len, mul_mat_vec_q5_K_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    vk_pipeline_dequant_mul_mat_vec_f32[GGML_TYPE_Q6_K] = ggml_vk_create_pipeline("mul_mat_vec_q6_K_f32", mul_mat_vec_q6_K_f32_len, mul_mat_vec_q6_K_f32_data, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);

    vk_pipeline_rms_norm_f32 = ggml_vk_create_pipeline("rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
}

void ggml_vk_test_transfer(size_t ne);
void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size);
void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size);
void ggml_vk_test_matmul_f16_f32(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size);

void ggml_vk_init(void) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_init()" << std::endl;
#endif
    const char* GGML_VULKAN_DEVICE = getenv("GGML_VULKAN_DEVICE");
    int dev_num = (GGML_VULKAN_DEVICE == NULL ? 0 : atoi(GGML_VULKAN_DEVICE));

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION };
    const std::vector<const char*> layers = {
#ifdef VK_VALIDATE
        "VK_LAYER_KHRONOS_validation",
#endif
    };
    const std::vector<const char*> extensions = {
#ifdef VK_VALIDATE
        "VK_EXT_validation_features",
#endif
    };
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers, extensions);
#ifdef VK_VALIDATE
    const std::vector<vk::ValidationFeatureEnableEXT> features_enable = { vk::ValidationFeatureEnableEXT::eBestPractices, vk::ValidationFeatureEnableEXT::eSynchronizationValidation };
    vk::ValidationFeaturesEXT validation_features = {
        features_enable,
        {},
    };
    validation_features.setPNext(nullptr);
    instance_create_info.setPNext(&validation_features);

std::cerr << "ggml_vulkan: Validation layers enabled" << std::endl;
#endif
    vk_instance = vk::createInstance(instance_create_info);

    vk_device.physical_device = vk_instance.enumeratePhysicalDevices()[dev_num];
    vk_device.properties = vk_device.physical_device.getProperties();
    std::cerr << "ggml_vulkan: Using " << vk_device.properties.deviceName << std::endl;

    vk_device.vendor_id = vk_device.properties.vendorID;

    std::vector<vk::ExtensionProperties> ext_props = vk_device.physical_device.enumerateDeviceExtensionProperties();

    bool fp16_storage = false;
    bool fp16_compute = false;

    for (auto properties : ext_props) {
        if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = true;
        } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = true;
        }
    }

    const char* GGML_VULKAN_DISABLE_F16 = getenv("GGML_VULKAN_DISABLE_F16");
    bool force_disable_f16 = GGML_VULKAN_DISABLE_F16 != NULL;

    vk_device.fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    std::vector<vk::QueueFamilyProperties> queue_family_props = vk_device.physical_device.getQueueFamilyProperties();

    // Try to find a non-graphics compute queue and transfer-focused queues
    const uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, -1, 1);
    const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics, compute_queue_family_index, 2);

    uint32_t transfer_queue_count = VK_TRANSFER_QUEUE_COUNT;

    // If not enough transfer queues are available
    if (transfer_queue_count > queue_family_props[transfer_queue_family_index].queueCount) {
        // If compute and transfer queues are same family
        if (compute_queue_family_index == transfer_queue_family_index) {
            transfer_queue_count = queue_family_props[transfer_queue_family_index].queueCount - 1;
        } else {
            transfer_queue_count = queue_family_props[transfer_queue_family_index].queueCount;
        }
    }

    const float compute_queue_priority = 1.0f;
    const float transfer_queue_priority[] = { 1.0f, 1.0f, 1.0f };
    std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
    if (compute_queue_family_index != transfer_queue_family_index) {
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, &compute_queue_priority});
        GGML_ASSERT(transfer_queue_count > 0);
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, transfer_queue_count, transfer_queue_priority});
    } else {
        device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, 1 + transfer_queue_count, transfer_queue_priority});
    }
    vk::DeviceCreateInfo device_create_info;
    std::vector<const char *> device_extensions;
    vk::PhysicalDeviceFeatures device_features = vk_device.physical_device.getFeatures();

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

    vkGetPhysicalDeviceFeatures2(vk_device.physical_device, &device_features2);

    vk_device.fp16 = vk_device.fp16 && vk12_features.shaderFloat16;

    if (!vk11_features.storageBuffer16BitAccess) {
        std::cerr << "ggml_vulkan: device does not support 16-bit storage" << std::endl;
        GGML_ASSERT(false);
    }

    device_extensions.push_back("VK_KHR_16bit_storage");

    if (vk_device.fp16) {
        std::cerr << "ggml_vulkan: 16-bit enabled" << std::endl;
        device_extensions.push_back("VK_KHR_shader_float16_int8");
    } else if (force_disable_f16) {
        std::cerr << "ggml_vulkan: 16-bit force-disabled" << std::endl;
    }
    device_create_info = {
        vk::DeviceCreateFlags(),
        device_queue_create_infos,
        {},
        device_extensions
    };
    device_create_info.setPNext(&device_features2);
    vk_device.device = vk_device.physical_device.createDevice(device_create_info);

    vk_device.descriptor_set_mode = VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN;

    // Shaders
    ggml_vk_load_shaders();

    // Queues
    uint32_t queue_index_offset = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;

    vk_device.compute_queue = ggml_vk_create_queue(compute_queue_family_index, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer });
    for (int i = 0; i < VK_TRANSFER_QUEUE_COUNT; i++) {
        if (transfer_queue_count > 0) {
            vk_device.transfer_queues[i] = ggml_vk_create_queue(transfer_queue_family_index, (queue_index_offset + i) % transfer_queue_count, { vk::PipelineStageFlagBits::eTransfer });
        } else {
            vk_device.transfer_queues[i] = vk_device.compute_queue;
        }
    }

    vk_fence = vk_device.device.createFence({});

#if defined(VK_CHK_KERNEL)
    int step = 16;
    for (size_t m = step; m < 64; m += step) {
        ggml_vk_test_transfer(1024 * 1024 * m);
    }
    const std::vector<size_t> vals {
        4096, 2, 4096,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
        11008, 49, 4096,
        4096, 49, 11008,
        32000, 49, 4096,
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
    };
    const size_t num_it = 100;
    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 0);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 0);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 1);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 1);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 2);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 2);
        std::cerr << std::endl;

        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 0);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 0);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 1);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 1);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 2);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 2);
        std::cerr << std::endl;

        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 0);
        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 0);
        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 1);
        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 1);
        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 1, 2);
        ggml_vk_test_matmul_f16_f32(vals[i], vals[i + 1], vals[i + 2], num_it, 4, 2);
        std::cerr << std::endl;
    }
#endif
}

static vk_pipeline* ggml_vk_get_to_fp16(ggml_type type) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_get_to_fp16()" << std::endl;
#endif
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return nullptr;
    }

    return &vk_pipeline_dequant[type];
}

static vk_pipeline* ggml_vk_get_dequantize_mul_mat_vec(ggml_type type, bool f16_y) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_get_dequantize_mul_mat_vec()" << std::endl;
#endif
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return nullptr;
    }

    return f16_y ? &vk_pipeline_dequant_mul_mat_vec[type] : &vk_pipeline_dequant_mul_mat_vec_f32[type];
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

static void ggml_vk_pool_malloc(size_t size, vk_buffer* buf, vk::MemoryPropertyFlags alloc_flags) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_pool_malloc(" << size << ", " << buf << ", " << to_string(alloc_flags) << ")" << std::endl;
#endif
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

    *buf = ggml_vk_create_buffer(size, vk::MemoryPropertyFlagBits::eDeviceLocal | alloc_flags);
}

static void ggml_vk_pool_free(vk_buffer& buffer) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_pool_free(" << buffer.size << ")" << std::endl;
#endif
    scoped_spin_lock lock(g_vk_pool_lock);

    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer& b = g_vk_buffer_pool[i];
        if (b.size == 0) {
            b = buffer;
            // Set owning queue family index to ignored to avoid synchronization on next use
            b.qf_owner = VK_QUEUE_FAMILY_IGNORED;
            return;
        }
    }
    fprintf(stderr, "WARNING: vk buffer pool full, increase MAX_VK_BUFFERS\n");
    ggml_vk_destroy_buffer(buffer);
}

void ggml_vk_free_data(const struct ggml_tensor* tensor) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_free_data(" << tensor << ")" << std::endl;
#endif
    if (tensor->backend != GGML_BACKEND_GPU) {
        return;
    }

    vk_buffer& buf = *(vk_buffer *)tensor->data;
    ggml_vk_destroy_buffer(buf);
    free(tensor->data);
}

void* ggml_vk_host_malloc(size_t size) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_host_malloc(" << size << ")" << std::endl;
#endif
    if (getenv("GGML_VK_NO_PINNED") != nullptr) {
        return nullptr;
    }

    vk_buffer buf = ggml_vk_create_buffer(size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached);

    if(!(buf.memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        buf.size = 0;
        vk_device.device.freeMemory(buf.device_memory);
        vk_device.device.destroyBuffer(buf.buffer);
        return nullptr;
    }

    vk_pinned_memory.push_back(std::make_tuple(buf.ptr, size, buf));

    return buf.ptr;
}

void ggml_vk_host_free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_host_free(" << ptr << ")" << std::endl;
#endif
    vk_buffer* buf = nullptr;
    size_t index;
    for (size_t i = 0; i < vk_pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(vk_pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(vk_pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = &std::get<2>(vk_pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(*buf);

    vk_pinned_memory.erase(vk_pinned_memory.begin() + index);
}

static vk_submission ggml_vk_begin_submission(vk_queue& q, bool one_time = true) {
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(q);
    if (one_time) {
        s.buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    } else {
        s.buffer.begin({ vk::CommandBufferUsageFlags{} });
    }

    return s;
}

static void ggml_vk_dispatch_pipeline(vk_submission& s, vk_pipeline& pipeline, std::vector<vk_subbuffer>&& buffers, size_t push_constant_size, const void* push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline.wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline.wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline.wg_denoms[2]);
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_dispatch_pipeline(" << pipeline.name << ", (" << wg0 << "," << wg1 << "," << wg2 << "))" << std::endl;
#endif
    std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    GGML_ASSERT(pipeline.descriptor_set_idx < pipeline.descriptor_sets.size());
    GGML_ASSERT(buffers.size() == pipeline.parameter_count);
    vk::DescriptorSet& descriptor_set = pipeline.descriptor_sets[pipeline.descriptor_set_idx++];
    for (uint32_t i = 0; i < pipeline.parameter_count; i++) {
        descriptor_buffer_infos.push_back({buffers[i].buffer.buffer, buffers[i].offset, buffers[i].size});
    }
    for (uint32_t i = 0; i < pipeline.parameter_count; i++) {
        write_descriptor_sets.push_back({descriptor_set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &descriptor_buffer_infos[i]});
    }

    vk_device.device.updateDescriptorSets(write_descriptor_sets, {});

    s.buffer.pushConstants(pipeline.layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size, push_constants);
    s.buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
    s.buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline.layout,
                                0,
                                { descriptor_set },
                                {});
    s.buffer.dispatch(wg0, wg1, wg2);
}

static void ggml_vk_end_submission(vk_submission& s, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
    s.buffer.end();

    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
}

static vk_sequence ggml_vk_buffer_write_2d_async(vk_buffer* dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")" << std::endl;
#endif
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ASSERT(false);
    }
    // Check if src is pinned memory
    vk_buffer* buf = nullptr;
    size_t buf_offset = 0;
    for (size_t i = 0; i < vk_pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(vk_pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(vk_pinned_memory[i]);
        if (src >= addr && src < endr) {
            buf = &std::get<2>(vk_pinned_memory[i]);
            buf_offset = ((const uint8_t *)src) - addr;
            break;
        }
    }

    bool reuse_submission = false;
    vk_submission submission;
    if (s == nullptr) {
        submission = ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores));
        s = &submission;
        reuse_submission = true;
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * width;
                slices[i].size = width;
            }
        }

        if (reuse_submission) {
            s->buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        }
        ggml_vk_sync_buffers(s->buffer, { ggml_vk_subbuffer(*dst) }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eMemoryWrite, false);
        s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
        if (reuse_submission) {
            s->buffer.end();
        }
        return { *s };
    }

    // Staging buffer required, malloc because of async transfer
    if (dst->sb_write == nullptr) {
        dst->sb_write = new vk_buffer;
        *dst->sb_write = ggml_vk_create_buffer(dst->size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    VkBufferCopy buf_copy = {
        0,
        offset,
        width * height};

    if (reuse_submission) {
        s->buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }
    ggml_vk_sync_buffers(s->buffer, { ggml_vk_subbuffer(*dst) }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eMemoryWrite, false);
    vkCmdCopyBuffer(s->buffer, dst->sb_write->buffer, dst->buffer, 1, &buf_copy);
    if (reuse_submission) {
        s->buffer.end();
    }

    if (width == spitch) {
        if (pre_staging == nullptr) {
            memcpy(dst->sb_write->ptr, src, width * height);
        } else {
            pre_staging->emplace_back((void *) dst->sb_write->ptr, (const void *) src, width * height);
        }
    } else {
        for (size_t i = 0; i < height; i++) {
            if (pre_staging == nullptr) {
                memcpy((uint8_t *)dst->sb_write->ptr + offset + i * width, (const uint8_t *) src + i * spitch, width);
            } else {
                pre_staging->emplace_back((void *) ((uint8_t *)dst->sb_write->ptr + offset + i * width), (const void *) ((const uint8_t *) src + i * spitch), width);
            }
        }
    }

    return { *s };
}

static void ggml_vk_buffer_write_2d(vk_buffer* dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_write_2d(" << width << ", " << height << ")" << std::endl;
#endif
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->ptr + offset + i * width, (const uint8_t *) src + i * spitch, width);
        }
    } else {
        vk::Fence fence = vk_device.device.createFence({});
        std::vector<vk_sequence> s = { ggml_vk_buffer_write_2d_async(dst, offset, src, spitch, width, height, q, {}, {}, nullptr) };
        ggml_vk_submit(q, s, fence);
        vk::resultCheck(vk_device.device.waitForFences({ fence }, true, uint64_t(-1)), "vk_buffer_write_2d waitForFences");
    }
}

static size_t ggml_vk_align_size(size_t width, size_t align) {
    return CEIL_DIV(width, align) * align;
}

static vk_sequence ggml_vk_buffer_write_async(vk_buffer* dst, size_t offset, const void * src, size_t size, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_write_async(" << size << ")" << std::endl;
#endif
    return ggml_vk_buffer_write_2d_async(dst, offset, src, 0, size, 1, q, std::move(wait_semaphores), std::move(signal_semaphores), s, pre_staging);
}

static void ggml_vk_buffer_write(vk_buffer* dst, size_t offset, const void * src, size_t size, vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_write(" << size << ")" << std::endl;
#endif
    ggml_vk_buffer_write_2d(dst, offset, src, 0, size, 1, q);
}

static vk_sequence ggml_vk_buffer_read_async(vk_buffer* src, size_t offset, void * dst, size_t size, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores, vk_submission* s = nullptr) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_read_async(" << size << ")" << std::endl;
#endif
    // Check if dst is pinned memory
    vk_buffer* buf = nullptr;
    size_t buf_offset = 0;
    for (size_t i = 0; i < vk_pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(vk_pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(vk_pinned_memory[i]);
        if (dst >= addr && dst < endr) {
            buf = &std::get<2>(vk_pinned_memory[i]);
            buf_offset = ((const uint8_t *)dst) - addr;
            break;
        }
    }

    if (buf == nullptr) {
        std::cerr << "ggml_vulkan: Error: buffer_read_async only works on pinned memory" << std::endl;
        GGML_ASSERT(false);
    }
    // Memory is pinned, use as staging buffer
    VkBufferCopy buf_copy = {
        offset, // srcOffset
        buf_offset, // dstOffset,
        size}; // size

    bool reuse_submission = false;
    vk_submission submission;
    if (s == nullptr) {
        submission = ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores));
        s = &submission;
        reuse_submission = true;
    }
    if (reuse_submission) {
        s->buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }
    ggml_vk_sync_buffers(s->buffer, { ggml_vk_subbuffer(*src) }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryRead, false);
    vkCmdCopyBuffer(s->buffer, src->buffer, buf->buffer, 1, &buf_copy);
    if (reuse_submission) {
        s->buffer.end();
    }

    return { *s };
}

static void ggml_vk_buffer_read(vk_buffer* src, size_t offset, void * dst, size_t size, vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_read(" << size << ")" << std::endl;
#endif
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        memcpy(dst, (uint8_t *) src->ptr + offset, size);
    } else {
        // Check if dst is pinned memory
        vk_buffer* buf = nullptr;
        size_t buf_offset = 0;
        for (size_t i = 0; i < vk_pinned_memory.size(); i++) {
            const uint8_t* addr = (const uint8_t*) std::get<0>(vk_pinned_memory[i]);
            const uint8_t* endr = addr + std::get<1>(vk_pinned_memory[i]);
            if (dst >= addr && dst < endr) {
                buf = &std::get<2>(vk_pinned_memory[i]);
                buf_offset = ((const uint8_t *)dst) - addr;
                break;
            }
        }

        if (buf != nullptr) {
            // Memory is pinned, use as staging buffer
            vk::Fence fence = vk_device.device.createFence({});
            VkBufferCopy buf_copy = {
                offset,
                buf_offset,
                size};

            std::vector<vk_sequence> s = { ggml_vk_create_sequence_1(q, {}, {}) };
            s[0][0].buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
            ggml_vk_sync_buffers(s[0][0].buffer, { ggml_vk_subbuffer(*src) }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead, false);
            vkCmdCopyBuffer(s[0][0].buffer, src->buffer, buf->buffer, 1, &buf_copy);
            s[0][0].buffer.end();
            ggml_vk_submit(q, s, fence);
            vk::resultCheck(vk_device.device.waitForFences({ fence }, true, uint64_t(-1)), "vk_buffer_read waitForFences");
            return;
        }

        if (src->sb_read == nullptr) {
            src->sb_read = new vk_buffer;
            *src->sb_read = ggml_vk_create_buffer(src->size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached);
        }

        VkBufferCopy buf_copy = {
            offset, // srcOffset
            0, // dstOffset,
            size}; // size

        vk::CommandBuffer cmd_buffer = ggml_vk_create_cmd_buffer(q);
        vk::CommandBufferBeginInfo cmd_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmd_buffer.begin(cmd_buffer_begin_info);
        ggml_vk_sync_buffers(cmd_buffer, { ggml_vk_subbuffer(*src) }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eTransferRead, false);
        vkCmdCopyBuffer(cmd_buffer, src->buffer, src->sb_read->buffer, 1, &buf_copy);
        cmd_buffer.end();

        vk::Fence fence = vk_device.device.createFence(vk::FenceCreateInfo{});

        vk::SubmitInfo submit_info(0,
                                   nullptr,
                                   nullptr,
                                   1,
                                   &cmd_buffer);
        q.queue.submit({ submit_info }, fence);
        vk::resultCheck(vk_device.device.waitForFences({ fence }, true, uint64_t(-1)), "vk_buffer_read staging waitForFences");
        vk_device.device.destroyFence(fence);
        memcpy(dst, src->sb_read->ptr, size);
    }
}

static vk_sequence ggml_vk_h2d_tensor_2d(vk_buffer* dst, size_t offset, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, uint64_t i1, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_h2d_tensor_2d()" << std::endl;
#endif
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
        return ggml_vk_buffer_write_async(dst, offset, x, i1*nb1, q, std::move(wait_semaphores), std::move(signal_semaphores), s, pre_staging);
    }
    if (nb0 == ts) {
        return ggml_vk_buffer_write_2d_async(dst, offset, x, nb1, row_length, i1, q, std::move(wait_semaphores), std::move(signal_semaphores), s, pre_staging);
    }
    GGML_ASSERT(false);
    // TODO: also needs handling of staging buffers
    uint8_t* dst_ptr = (uint8_t*) dst->ptr;
    const uint8_t* xc = (const uint8_t*)x;
    for (uint64_t i1 = 0; i1 < ne1; i1++) {
        for (uint64_t i0 = 0; i0 < ne0; i0++) {
            dst_ptr[offset + i1 * row_length + i0 * ts] = xc[i1 * nb1 + i0 * nb0];
        }
    }
}

static int ggml_vk_guess_split_k(int m, int n, int k, bool aligned) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ", " << aligned << ")";
#endif
    if (aligned && k > 128 && (m < 128 || n < 128)) {
#ifdef VK_DEBUG
    std::cerr << " = 4" << std::endl;
#endif
        return 4;
    }

#ifdef VK_DEBUG
    std::cerr << " = 1" << std::endl;
#endif
    return 1;
}

static uint32_t ggml_vk_guess_matmul_pipeline_align(int m, int n) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ")" << std::endl;
#endif
    if (m <= 32 || n <= 32) {
        return vk_pipeline_matmul_f32_aligned_s.align;
    }
    if (m <= 64 || n <= 64) {
        return vk_pipeline_matmul_f32_aligned_m.align;
    }
    return vk_pipeline_matmul_f32_aligned_l.align;
}

static vk_pipeline* ggml_vk_guess_matmul_pipeline(bool bit16_x, bool bit16_y, int m, int n, bool aligned) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_guess_matmul_pipeline(" << bit16_x << ", " << bit16_y << ", " << m << ", " << n << ", " << aligned << ")";
#endif
    if (bit16_x && bit16_y) {
        if (m <= 32 || n <= 32) {
#ifdef VK_DEBUG
    std::cerr << " S" << std::endl;
#endif
            return aligned ? &vk_pipeline_matmul_f16_aligned_s : &vk_pipeline_matmul_f16_s;
        }
        if (m <= 64 || n <= 64) {
#ifdef VK_DEBUG
    std::cerr << " M" << std::endl;
#endif
            return aligned ? &vk_pipeline_matmul_f16_aligned_m : &vk_pipeline_matmul_f16_m;
        }
#ifdef VK_DEBUG
    std::cerr << " L" << std::endl;
#endif
        return aligned ? &vk_pipeline_matmul_f16_aligned_l : &vk_pipeline_matmul_f16_l;
    }
    if (bit16_x && !bit16_y) {
        if (m <= 32 || n <= 32) {
#ifdef VK_DEBUG
    std::cerr << " S" << std::endl;
#endif
            return aligned ? &vk_pipeline_matmul_f16_f32_aligned_s : &vk_pipeline_matmul_f16_f32_s;
        }
        if (m <= 64 || n <= 64) {
#ifdef VK_DEBUG
    std::cerr << " M" << std::endl;
#endif
            return aligned ? &vk_pipeline_matmul_f16_f32_aligned_m : &vk_pipeline_matmul_f16_f32_m;
        }
#ifdef VK_DEBUG
    std::cerr << " L" << std::endl;
#endif
        return aligned ? &vk_pipeline_matmul_f16_f32_aligned_l : &vk_pipeline_matmul_f16_f32_l;
    }
    if (!bit16_x && bit16_y) {
        GGML_ASSERT(false);
    }

    if (m <= 32 || n <= 32) {
#ifdef VK_DEBUG
    std::cerr << " S" << std::endl;
#endif
        return aligned ? &vk_pipeline_matmul_f32_aligned_s : &vk_pipeline_matmul_f32_s;
    }
    if (m <= 64 || n <= 64) {
#ifdef VK_DEBUG
    std::cerr << " M" << std::endl;
#endif
        return aligned ? &vk_pipeline_matmul_f32_aligned_m : &vk_pipeline_matmul_f32_m;
    }
#ifdef VK_DEBUG
    std::cerr << " L" << std::endl;
#endif
    return aligned ? &vk_pipeline_matmul_f32_aligned_l : &vk_pipeline_matmul_f32_l;
}

static vk_sequence ggml_vk_matmul(vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, int m, int n, int k, int stride_a, int stride_b, int stride_d, int split_k, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_matmul(" << m << ", " << n << ", " << k << ")" << std::endl;
#endif
    vk_submission s = ggml_vk_begin_submission(q);
    ggml_vk_sync_buffers(s.buffer, { a, b }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead, false);
    ggml_vk_sync_buffers(s.buffer, { d }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eShaderWrite, false);
    if (split_k == 1) {
        const std::vector<int> pc = { m, n, k, stride_a, stride_b, stride_d, k };
        ggml_vk_dispatch_pipeline(s, pipeline, { a, b, d }, pc.size() * sizeof(int), pc.data(), { (uint32_t)m, (uint32_t)n, 1 });
        ggml_vk_end_submission(s, std::move(wait_semaphores), std::move(signal_semaphores));
        return { s };
    }

    // Synchronize the two submissions
    const std::vector<int> pc1 = { m, n, k, stride_a, stride_b, stride_d, CEIL_DIV(k, split_k) };
    ggml_vk_dispatch_pipeline(s, pipeline, { a, b, d }, pc1.size() * sizeof(int), pc1.data(), { (uint32_t)m * split_k, (uint32_t)n, 1 });
    ggml_vk_sync_buffers(s.buffer, { d }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, true);
    const std::vector<int> pc2 = { m, n, split_k };
    ggml_vk_dispatch_pipeline(s, vk_pipeline_matmul_split_k_reduce, { d }, pc2.size() * sizeof(int), pc2.data(), { (uint32_t)m, (uint32_t)n, 1 });
    ggml_vk_end_submission(s, std::move(wait_semaphores), std::move(signal_semaphores));

    return { s };
}

static void ggml_vk_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_f32((" << src0 << ", type=" << src0->type << ",  backend=" << src0->backend << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", type=" << src1->type << ",  backend=" << src1->backend << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", type=" << dst->type << ",  backend=" << dst->backend << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)" << std::endl;
#endif
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const int kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ne01, ne11));
    const bool aligned = ne10 == kpad;

    const int split_k = ggml_vk_guess_split_k(ne01, ne11, ne10, aligned);

    const bool load_x = src0->backend == GGML_BACKEND_GPU;

    vk_pipeline * pipeline = ggml_vk_guess_matmul_pipeline(false, false, ne01, ne11, aligned);

    const uint32_t x_sz = ggml_vk_align_size(sizeof(float) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(sizeof(float) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    vk_buffer* d_D;
    if (dst->backend == GGML_BACKEND_GPU) {
        d_D = (vk_buffer *) dst->data;
    } else {
        d_D = &vk_prealloc_d;
    }
    vk_buffer* d_X;
    vk_buffer* d_Y;
    if (load_x) {
        d_X = &vk_prealloc_qx;
    } else {
        d_X = (vk_buffer *) src0->data;
    }
    d_Y = &vk_prealloc_qy;

    // Allocate descriptor sets
    ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne12 * ne13);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, ne12 * ne13);
    }

    std::vector<vk_semaphore> x_semaphores;

    if (load_x) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const uint32_t x_offset = load_x ? x_sz * (i03 * ne02 + i02) : 0;
                // copy data to device
                vk_semaphore * sem = ggml_vk_create_timeline_semaphore();
                x_semaphores.push_back({ sem->s, sem->value + 1 });
                // Wait for previous matmul to be done before writing to the input buffers again
                extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_X, x_offset, src0, i03, i02, ne01, vk_device.transfer_queues[0], {}, { { sem->s, sem->value + 1 } }, nullptr, &extra->memcpys));
                sem->value += 1;
            }
        }
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        const int64_t i03 = i13 / r3;
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 / r2;

            const uint32_t x_offset = load_x ? x_sz * (i03 * ne02 + i02) : 0;
            const uint32_t y_offset = y_sz * (i13 * ne12 + i12);
            const uint32_t d_offset = d_sz * (i13 * ne12 + i12);

            vk_semaphore * sem = ggml_vk_create_timeline_semaphore();
            std::vector<vk_semaphore> semaphores = { { sem->s, sem->value + 1 } };

            if (load_x) {
                semaphores.push_back(x_semaphores[i03 * ne02 + i02]);
            }

            extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Y, y_offset, src1, i13, i12, ne11, vk_device.transfer_queues[1], {}, { { sem->s, sem->value + 1 } }, nullptr, &extra->memcpys));

            // compute
            extra->comp_seqs.push_back(ggml_vk_matmul(*pipeline, { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz }, ne01, ne11, ne10, ne10, ne10, ne01, split_k, vk_device.compute_queue, std::move(semaphores), { { sem->s, sem->value + 2 } }));

            if (dst->backend == GGML_BACKEND_CPU) {
                // copy dst to host
                float * d = (float *) ((char *) dst->data + i12*nb2 + i13*nb3);
                extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, vk_device.transfer_queues[1], { { sem->s, sem->value + 2 } }, {}));
            }

            sem->value += 2;
        }
    }
}

static void ggml_vk_mul_mat_q_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_q_f16((" << src0 << ", type=" << src0->type << ",  backend=" << src0->backend << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", type=" << src1->type << ",  backend=" << src1->backend << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", type=" << dst->type << ",  backend=" << dst->backend << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)" << std::endl;
#endif
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    vk_queue& compq = vk_device.compute_queue;
    vk_queue& tr0q = vk_device.transfer_queues[0];
    vk_queue& tr1q = vk_device.transfer_queues[1];
    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = src0->type != GGML_TYPE_F16;
    const bool qy_needs_dequant = src1->type != GGML_TYPE_F16 && !f16_f32_kernel;

    const bool load_x = src0->backend != GGML_BACKEND_GPU;
    const bool load_y = src1->backend != GGML_BACKEND_GPU;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const int kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ne01, ne11));
    const bool aligned = ne10 == kpad;

    const int split_k = ggml_vk_guess_split_k(ne01, ne11, ne10, aligned);

    vk_pipeline * pipeline = ggml_vk_guess_matmul_pipeline(true, !f16_f32_kernel, ne01, ne11, aligned);

    const uint32_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t qy_sz = ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t x_sz = ggml_vk_align_size(sizeof(ggml_fp16_t) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    if (dst->backend == GGML_BACKEND_GPU) {
        if (d_sz != nb2) {
            std::cerr << "ERROR: incompatible tensor alignment d_sz=" << d_sz << " nb2=" << nb2 << std::endl;
            GGML_ASSERT(false);
        }
    }

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    vk_buffer* d_D;
    if (dst->backend == GGML_BACKEND_GPU) {
        d_D = (vk_buffer *) dst->data;
    } else {
        d_D = &vk_prealloc_d;
    }
    GGML_ASSERT(d_D->size >= d_sz * ne02 * ne03);
    vk_buffer* d_Qx;
    vk_buffer* d_Qy;
    vk_buffer* d_X;
    vk_buffer* d_Y;
    if (load_x) {
        d_Qx = &vk_prealloc_qx;
        GGML_ASSERT(d_Qx->size >= qx_sz * ne02 * ne03);
    } else {
        d_Qx = (vk_buffer *) src0->data;
    }
    if (load_y) {
        d_Qy = &vk_prealloc_qy;
        GGML_ASSERT(d_Qy->size >= qy_sz * ne02 * ne03);
    } else {
        d_Qy = (vk_buffer *) src1->data;
    }
    if (qx_needs_dequant) {
        d_X = &vk_prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz * ne02 * ne03);
    } else {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);  // NOLINT
    }
    if (qy_needs_dequant) {
        d_Y = &vk_prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz * ne02 * ne03);
    } else {
        d_Y = d_Qy;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline* to_fp16_vk_0 = ggml_vk_get_to_fp16(src0->type);
    vk_pipeline* to_fp16_vk_1 = ggml_vk_get_to_fp16(src1->type);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    // Allocate descriptor sets
    ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne12 * ne13);
    if (qx_needs_dequant) {
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_0, ne12 * ne13);
    }
    if (qy_needs_dequant) {
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_1, ne12 * ne13);
    }
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, ne12 * ne13);
    }

    std::vector<vk_semaphore> x_semaphores;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint32_t it_idx0 = (i03 * ne02 + i02);
            const uint32_t qx_offset = qx_sz * it_idx0;
            const uint32_t x_offset = x_sz * it_idx0;

            vk_semaphore * sem = ggml_vk_create_timeline_semaphore();

            if (load_x) {
                // copy data to device
                extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Qx, qx_offset, src0, i03, i02, ne01, tr0q, {}, { { sem->s, sem->value + 1 } }, nullptr, &extra->memcpys));
            }

            if (qx_needs_dequant) {
                vk_submission s = ggml_vk_begin_submission(compq);
                const std::vector<int> pc = { (int)ne01, (int)ne10, (int)ne10, (int)ne10 };
                ggml_vk_sync_buffers(s.buffer, { { *d_Qx, qx_offset, qx_sz } }, compq, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                ggml_vk_sync_buffers(s.buffer, { { *d_X, x_offset, x_sz } }, compq, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite, false);
                ggml_vk_dispatch_pipeline(s, *to_fp16_vk_0, { { *d_Qx, qx_offset, qx_sz }, { *d_X, x_offset, x_sz } }, pc.size() * sizeof(int), pc.data(), { (uint32_t)x_ne, 1, 1});
                if (load_x) {
                    ggml_vk_end_submission(s, { { sem->s, sem->value + 1 } }, { { sem->s, sem->value + 2 } });
                } else {
                    ggml_vk_end_submission(s, {}, { { sem->s, sem->value + 2 } });
                }

                extra->comp_seqs.push_back({ s });

                x_semaphores.push_back({ sem->s, sem->value + 2 });
            } else if (load_x) {
                x_semaphores.push_back({ sem->s, sem->value + 1 });
            } else {
                x_semaphores.push_back({ sem->s, sem->value });
            }

            sem->value += 2;
        }
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        const int64_t i03 = i13 / r3;
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 / r2;

            const uint32_t it_idx0 = (i03 * ne02 + i02);
            const uint32_t it_idx1 = (i13 * ne12 + i12);
            const uint32_t qy_offset = load_y ? qy_sz * it_idx1 : 0;
            const uint32_t x_offset = x_sz * it_idx0;
            const uint32_t y_offset = y_sz * it_idx1;
            const uint32_t d_offset = d_sz * it_idx1;

            vk_semaphore * sem = ggml_vk_create_timeline_semaphore();

            std::vector<vk_semaphore> mm_semaphores;

            if (load_x) {
                mm_semaphores.push_back(x_semaphores[it_idx0]);
            }
            if (load_y) {
                // Set semaphore to 1
                extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Qy, qy_offset, src1, i13, i12, ne11, tr1q, {}, { { sem->s, sem->value + 1 } }, nullptr, &extra->memcpys));
                // Wait for semaphore val 1
                mm_semaphores.push_back({ sem->s, sem->value + 1 });
            }

            // compute
            extra->comp_seqs.push_back(ggml_vk_matmul(*pipeline, { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz }, ne01, ne11, ne10, ne10, ne10, ne01, split_k, compq, std::move(mm_semaphores), { { sem->s, sem->value + 2 } }));

            if (dst->backend == GGML_BACKEND_CPU) {
                // copy dst to host
                float * d = (float *) ((char *) dst->data + i12*nb2 + i13*nb3);
                extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, tr1q, { { sem->s, sem->value + 2 } }, {}));
            }

            sem->value += 2;
        }
    }
}

static void ggml_vk_mul_mat_vec_q_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_vec_q_f16((" << src0 << ", type=" << src0->type << ",  backend=" << src0->backend << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", type=" << src1->type << ",  backend=" << src1->backend << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", type=" << dst->type << ",  backend=" << dst->backend << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)" << std::endl;
#endif
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);
    GGML_ASSERT(src0->backend == GGML_BACKEND_GPU);

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    vk_queue& compq = vk_device.compute_queue;
    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qy_needs_dequant = src1->type != GGML_TYPE_F16 && !f16_f32_kernel;

    const bool load_y = src1->backend != GGML_BACKEND_GPU;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const uint32_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t qy_sz = ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    GGML_ASSERT(qy_sz == src1->nb[2]);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    vk_buffer* d_D;
    if (dst->backend == GGML_BACKEND_GPU) {
        d_D = (vk_buffer *) dst->data;
    } else {
        d_D = &vk_prealloc_d;
    }
    vk_buffer* d_Qx;
    vk_buffer* d_Qy;
    vk_buffer* d_Y;
    d_Qx = (vk_buffer *) src0->data;
    if (load_y) {
        d_Qy = &vk_prealloc_qy;
    } else {
        d_Qy = (vk_buffer *) src1->data;
    }
    if (qy_needs_dequant) {
        d_Y = &vk_prealloc_y;
    } else {
        d_Y = d_Qy;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline* to_fp16_vk_1 = ggml_vk_get_to_fp16(src1->type);
    vk_pipeline* dmmv = ggml_vk_get_dequantize_mul_mat_vec(src0->type, !f16_f32_kernel);
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    // Allocate descriptor sets
    if (qy_needs_dequant) {
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_1, ne12 * ne13);
    }
    ggml_vk_pipeline_allocate_descriptor_sets(*dmmv, ne12 * ne13);

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        const int64_t i03 = i13 / r3;
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 / r2;

            const uint32_t it_idx0 = (i03 * ne02 + i02);
            const uint32_t it_idx1 = (i13 * ne12 + i12);
            const uint32_t qx_offset = qx_sz * it_idx0;
            const uint32_t qy_offset = qy_sz * it_idx1;
            const uint32_t y_offset = y_sz * it_idx1;
            const uint32_t d_offset = d_sz * it_idx1;

            vk_submission s = ggml_vk_begin_submission(compq);

            vk_semaphore s_x;
            if (load_y) {
                ggml_vk_h2d_tensor_2d(d_Qy, qy_offset, src1, i13, i12, ne11, compq, {}, {}, &s, &extra->memcpys);
            }

            if (qy_needs_dequant) {
                const std::vector<int> pc = { (int)ne11, (int)ne10, (int)ne10, (int)ne10 };
                ggml_vk_sync_buffers(s.buffer, { { *d_Qy, qy_offset, qy_sz } }, compq, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, true);
                ggml_vk_sync_buffers(s.buffer, { { *d_Y, y_offset, y_sz } }, compq, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, false);
                ggml_vk_dispatch_pipeline(s, *to_fp16_vk_1, { { *d_Qy, qy_offset, qy_sz }, { *d_Y, y_offset, y_sz } }, pc.size() * sizeof(int), pc.data(), { (uint32_t)y_ne, 1, 1});
            }

            // compute
            const int ncols = ne00;
            ggml_vk_sync_buffers(s.buffer, { { *d_Qx, qx_offset, qx_sz }, { *d_Y, y_offset, y_sz } }, compq, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, true);
            ggml_vk_sync_buffers(s.buffer, { { *d_D, d_offset, d_sz } }, compq, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, false);
            ggml_vk_dispatch_pipeline(s, *dmmv, { { *d_Qx, qx_offset, qx_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz } }, sizeof(int), &ncols, { (uint32_t)ne01, 1, 1});

            if (dst->backend == GGML_BACKEND_CPU) {
                // copy dst to host
                float * d = (float *) ((char *) dst->data + i12*nb2 + i13*nb3);
                ggml_vk_sync_buffers(s.buffer, { { *d_D, d_offset, d_sz } }, compq, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead, true);
                ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, compq, {}, {}, &s);
            }

            ggml_vk_end_submission(s, {}, {});

            extra->comp_seqs.push_back({ s });
        }
    }
}

static bool ggml_vk_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || ggml_is_quantized(src1->type)) &&
        dst->type == GGML_TYPE_F16 &&
        ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32) || src0->backend == GGML_BACKEND_GPU)) {
        std::cerr << "FP16 dst required" << std::endl;
    }

    // TODO: find the optimal values for these
    if ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
        (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || ggml_is_quantized(src1->type)) &&
        dst->type == GGML_TYPE_F32 &&
        ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32) || src0->backend == GGML_BACKEND_GPU)) {
        return true;
    }

    return false;
}

static void ggml_vk_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")" << std::endl;
#endif
    GGML_ASSERT(ggml_vk_can_mul_mat(src0, src1, dst));

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_vk_mul_mat_f32(src0, src1, dst);
    } else if (src1->ne[1] == 1 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(src0, src1, dst);
    } else {
        ggml_vk_mul_mat_q_f16(src0, src1, dst);
    }
}

static void ggml_vk_op_repeat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // guaranteed to be an integer due to the check in ggml_can_repeat
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(src0->backend == GGML_BACKEND_GPU);
    GGML_ASSERT(dst->backend == GGML_BACKEND_GPU);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    const vk_buffer* src0_buf = (vk_buffer *) src0->data;
    vk_buffer* dst_buf = (vk_buffer *) dst->data;

    std::vector<VkBufferCopy> copies;

    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                copies.push_back({
                                    (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0,
                                    (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01,
                                    ne00*nb0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    vk_submission s = ggml_vk_begin_submission(vk_device.transfer_queues[0]);
    vkCmdCopyBuffer(s.buffer, src0_buf->buffer, dst_buf->buffer, copies.size(), copies.data());
    ggml_vk_end_submission(s, {}, {});
    extra->out_seqs.push_back({ s });

    (void) src1;
}


static vk_pipeline* ggml_vk_op_get_pipeline(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_ADD:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_add_f32;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return &vk_pipeline_add_f16_f32_f16;
        }
        return nullptr;
    case GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        if (dst->type == GGML_TYPE_F16) {
            return &vk_pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_get_rows_f32[src0->type];
        }
        return nullptr;
    case GGML_OP_MUL:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_mul_f32;
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_scale_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_rms_norm_f32;
        }
        return nullptr;
    default:
        return nullptr;
    }
}

static ggml_vk_func_t ggml_vk_op_get_func(ggml_op op) {
    switch(op) {
    case GGML_OP_REPEAT:
        return ggml_vk_op_repeat;
    default:
        return nullptr;
    }
}

static bool ggml_vk_dim01_contiguous(const ggml_tensor * tensor) {
    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static void ggml_vk_op_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_op op, float param=1.0f) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_op_f32((" << src0 << ", type=" << src0->type << ",  backend=" << src0->backend << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", type=" << src1->type << ",  backend=" << src1->backend << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    std::cerr << "), (" << dst << ", type=" << dst->type << ",  backend=" << dst->backend << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "), " << ggml_op_name(op) << ")" << std::endl;
#endif
    GGML_ASSERT(!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type)));  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0));
    GGML_ASSERT(src1 == nullptr || ggml_vk_dim01_contiguous(src1));  // NOLINT
    GGML_ASSERT(dst->extra != nullptr);
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne0 = ne00 * ne01;
    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const int64_t ne1 = ne10 * ne11;
    const int64_t nb10 = use_src1 ? src1->nb[0] : 0;
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    GGML_ASSERT(dst->ne[0] * dst->ne[1] == ne0);
    GGML_ASSERT(!use_src1 || nb10 == sizeof(float));  // NOLINT

    vk_pipeline * pipeline = ggml_vk_op_get_pipeline(src0, src1, dst, op);
    ggml_vk_func_t op_func;

    if (pipeline == nullptr) {
        op_func = ggml_vk_op_get_func(op);
        GGML_ASSERT(op_func != nullptr);

        op_func(src0, src1, dst);
        return;
    }

    const bool transfer_src0 = src0->backend != GGML_BACKEND_GPU;
    const bool transfer_src1 = use_src1 && src1->backend != GGML_BACKEND_GPU;

    const uint32_t x_sz = ggml_vk_align_size(ggml_type_size(src0->type) * ne0, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = use_src1 ? ggml_vk_align_size(ggml_type_size(src1->type) * ne1, vk_device.properties.limits.minStorageBufferOffsetAlignment) : 0;
    const uint32_t d_sz = ggml_vk_align_size(ggml_type_size(dst->type) * ne0, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    vk_buffer* d_D;
    if (dst->backend == GGML_BACKEND_GPU) {
        d_D = (vk_buffer *) dst->data;
    } else {
        d_D = &vk_prealloc_d;
    }
    vk_buffer* d_X = nullptr;
    vk_buffer* d_Y = nullptr;
    if (transfer_src0) {
        d_X = &vk_prealloc_qx;
    } else {
        d_X = (vk_buffer *) src0->data;
    }
    if (transfer_src1) {
        d_Y = &vk_prealloc_qy;
    } else if (use_src1) {
        d_Y = (vk_buffer *) src1->data;
    }

    vk_op_push_constants pc;
    std::array<uint32_t, 3> elements;

    std::vector<vk_semaphore> transfer_semaphores;
    // copy src0 to device
    if (transfer_src0) {
        vk_semaphore * sem_x = ggml_vk_create_timeline_semaphore();
        extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_X, 0, src0, 0, 0, ggml_nrows(src0), vk_device.transfer_queues[0], {}, { { sem_x->s, sem_x->value + 1 } }, nullptr, &extra->memcpys));
        transfer_semaphores.push_back({ sem_x->s, sem_x->value + 1});
        sem_x->value += 1;
    }
    if (transfer_src1) {
        vk_semaphore * sem_y = ggml_vk_create_timeline_semaphore();
        extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Y, 0, src1, 0, 0, ggml_nrows(src1), vk_device.transfer_queues[1], {}, { { sem_y->s, sem_y->value + 1 } }, nullptr, &extra->memcpys));
        transfer_semaphores.push_back({ sem_y->s, sem_y->value + 1 });
        sem_y->value += 1;
    }

    // Single call if dimension 2 is contiguous
    if (ggml_is_contiguous(src0) && (src1 == nullptr || ggml_is_contiguous(src1))) {
        ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, 1);

        switch (dst->op) {
        case GGML_OP_RMS_NORM:
            pc = { (int)src0->ne[0], (int)src0->ne[1], param };
            elements = { (uint32_t)ggml_nrows(src0), 1, 1 };
            break;
        default:
            pc = { (int)ggml_nelements(src0), (int)(src1 != nullptr ? ggml_nelements(src1) : 0), param };
            elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
            break;
        }

        vk_submission s = ggml_vk_begin_submission(vk_device.compute_queue);
        ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_D) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, false);
        if (use_src1) {
            ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_X), ggml_vk_subbuffer(*d_Y) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
            ggml_vk_dispatch_pipeline(s, *pipeline, { ggml_vk_subbuffer(*d_X), ggml_vk_subbuffer(*d_Y), ggml_vk_subbuffer(*d_D) }, sizeof(vk_op_push_constants), &pc, elements);
        } else {
            ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_X) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
            ggml_vk_dispatch_pipeline(s, *pipeline, { ggml_vk_subbuffer(*d_X), ggml_vk_subbuffer(*d_D) }, sizeof(vk_op_push_constants), &pc, elements);
        }
        if (dst->backend == GGML_BACKEND_CPU) {
            vk_semaphore * fsem = ggml_vk_create_binary_semaphore();
            ggml_vk_end_submission(s, std::move(transfer_semaphores), { *fsem });
            extra->comp_seqs.push_back({ s });

            // copy dst to host
            float * d = (float *) dst->data;
            extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, 0, d, sizeof(float) * ggml_nelements(src0), vk_device.transfer_queues[1], { *fsem }, {}));
        } else {
            ggml_vk_end_submission(s, std::move(transfer_semaphores), {});
            extra->comp_seqs.push_back({ s });
        }
    } else {
        ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne02 * ne03);

        switch (dst->op) {
        case GGML_OP_RMS_NORM:
            pc = { (int)src0->ne[0], (int)src0->ne[1], param };
            elements = { (uint32_t)ne01, 1, 1 };
            break;
        default:
            pc = { (int)ne0, (int)(src1 != nullptr ? ne1 : 0), param };
            elements = { (uint32_t)ne0, 1, 1 };
            break;
        }

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const uint32_t it_idx0 = (i03 * ne02 + i02);
                const uint32_t it_idx1 = use_src1 ? ((i03 % ne13) * ne12 + (i02 % ne12)) : 0;
                const uint32_t x_offset = x_sz * it_idx0;
                const uint32_t y_offset = y_sz * it_idx1;
                const uint32_t d_offset = d_sz * it_idx0;

                vk_submission s = ggml_vk_begin_submission(vk_device.compute_queue);
                ggml_vk_sync_buffers(s.buffer, { { *d_D, d_offset, d_sz } }, vk_device.compute_queue, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, false);
                if (use_src1) {
                    ggml_vk_sync_buffers(s.buffer, { { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz } }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                    ggml_vk_dispatch_pipeline(s, *pipeline, { { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz } }, sizeof(vk_op_push_constants), &pc, elements);
                } else {
                    ggml_vk_sync_buffers(s.buffer, { { *d_X, x_offset, x_sz } }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                    ggml_vk_dispatch_pipeline(s, *pipeline, { { *d_X, x_offset, x_sz }, { *d_D, d_offset, d_sz } }, sizeof(vk_op_push_constants), &pc, elements);
                }
                if (dst->backend == GGML_BACKEND_CPU) {
                    vk_semaphore * fsem = ggml_vk_create_binary_semaphore();
                    ggml_vk_end_submission(s, std::move(transfer_semaphores), { *fsem });
                    extra->comp_seqs.push_back({ s });

                    // copy dst to host
                    extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, (char *) dst->data + i02*nb2 + i03*nb3, sizeof(float) * ne0, vk_device.transfer_queues[1], { *fsem }, {}));
                } else {
                    ggml_vk_end_submission(s, std::move(transfer_semaphores), {});
                    extra->comp_seqs.push_back({ s });
                }
            }
        }
    }
}

static void ggml_vk_repeat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_REPEAT);
}

static void ggml_vk_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_GET_ROWS);
}

static void ggml_vk_add(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_ADD);
}

static void ggml_vk_mul(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_MUL);
}

static void ggml_vk_scale(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, nullptr, dst, GGML_OP_SCALE, ((float *)src1->data)[0]);
}

static void ggml_vk_rms_norm(const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(src0, nullptr, dst, GGML_OP_RMS_NORM, ((float *)src0->op_params)[0]);
}

void ggml_vk_transform_tensor(void * data, ggml_tensor * tensor) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_transform_tensor(" << data << ", " << tensor << ")" << std::endl;
#endif
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];

    GGML_ASSERT(ne2 == 1 && ne3 == 1);  // NOLINT

    const ggml_type type = tensor->type;
    const size_t q_sz = ggml_type_size(type) * ne0 * ne1 * ne2 * ne3 / ggml_blck_size(type);

    vk_buffer dst = ggml_vk_create_buffer(q_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);

    std::vector<vk_sequence> seqs;

    tensor->data = data;
    // copy tensor to device
    seqs.push_back(ggml_vk_h2d_tensor_2d(&dst, 0, tensor, 0, 0, ne1, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seqs, VK_NULL_HANDLE);
    vk_device.transfer_queues[0].queue.waitIdle();

    tensor->data = malloc(sizeof(vk_buffer));
    *(vk_buffer*) tensor->data = dst;
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
}

static void ggml_vk_realign_tensor(ggml_tensor * tensor) {
    // Handle split-k, which needs more space per MM
    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) tensor->extra;

    tensor->nb[2] = ggml_vk_align_size(std::max(extra->tensor_size / tensor->ne[3] / tensor->ne[2], tensor->nb[1]*tensor->ne[1]), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    for (int i = 3; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }
}

static ggml_vk_tensor_extra_gpu * ggml_vk_preallocate_buffers(uint32_t d_size, uint32_t qx_size, uint32_t qy_size, uint32_t x_size, uint32_t y_size) {
    ggml_vk_tensor_extra_gpu * extra = new ggml_vk_tensor_extra_gpu;

    extra->tensor_size = d_size;
    extra->gpu_buffer = nullptr;

    // Check if buffer already exists, increase size if required
    if (vk_prealloc_size_d < d_size) {
        vk_prealloc_size_d = d_size;
    }
    if (vk_prealloc_size_qx < qx_size) {
        vk_prealloc_size_qx = qx_size;
    }
    if (vk_prealloc_size_qy < qy_size) {
        vk_prealloc_size_qy = qy_size;
    }
    if (vk_prealloc_size_x < x_size) {
        vk_prealloc_size_x = x_size;
    }
    if (vk_prealloc_size_y < y_size) {
        vk_prealloc_size_y = y_size;
    }

    vk_gc.extras.push_back(extra);

    return extra;
}

void ggml_vk_preallocate_buffers_graph(ggml_tensor * node){
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_preallocate_buffers_graph(" << node << ")" << std::endl;
#endif
    node->extra = nullptr;

    const bool src0_gpu = false;  // node->src[0] != nullptr && node->src[0]->ne[1] > 32 && node->src[0]->extra != nullptr && node->src[0]->backend == GGML_BACKEND_CPU;
    const bool src1_gpu = false;  // node->src[1] != nullptr && node->src[1]->ne[1] > 32 && node->src[1]->extra != nullptr && node->src[1]->backend == GGML_BACKEND_CPU;

    const bool any_on_device = node->backend == GGML_BACKEND_GPU
        || (node->src[0] != nullptr && (node->src[0]->backend == GGML_BACKEND_GPU || node->src[0]->backend == GGML_BACKEND_GPU_SPLIT || src0_gpu))
        || (node->src[1] != nullptr && (node->src[1]->backend == GGML_BACKEND_GPU || src1_gpu));

    if (!any_on_device && node->op != GGML_OP_MUL_MAT) {
        return;
    }

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    const bool use_src0 = src0 != nullptr;
    const int64_t ne00 = use_src0 ? src0->ne[0] : 0;
    const int64_t ne01 = use_src0 ? src0->ne[1] : 0;
    const int64_t ne02 = use_src0 ? src0->ne[2] : 0;
    const int64_t ne03 = use_src0 ? src0->ne[3] : 0;
    const bool use_src1 = src1 != nullptr && node->op != GGML_OP_SCALE;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const int64_t ne20 = node->ne[0];
    const int64_t ne21 = node->ne[1];
    const int64_t ne22 = node->ne[2];
    const int64_t ne23 = node->ne[3];

    const bool transfer_src0 = use_src0 && src0->backend != GGML_BACKEND_GPU;
    const bool transfer_src1 = use_src1 && src1->backend != GGML_BACKEND_GPU;

    const bool qvec_kernel = use_src0 && use_src1 && src1->ne[1] == 1 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
    const bool qx_needs_dequant = use_src0 && !qvec_kernel && src0->type != GGML_TYPE_F16;
    const bool f16_f32_kernel = use_src1 && src1->type == GGML_TYPE_F32;
    const bool qy_needs_dequant = use_src1 && src1->type != GGML_TYPE_F16 && !f16_f32_kernel;

    int split_k;
    if (node->op == GGML_OP_MUL_MAT) {
        const int kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ne01, ne11));
        const bool aligned = ne10 == kpad;
        split_k = ggml_vk_guess_split_k(ne01, ne11, ne10, aligned);
    } else {
        split_k = 1;
    }
    const uint32_t x_ne = ne00 * ne01;
    const uint32_t y_ne = ne10 * ne11;
    const uint32_t d_ne = ne20 * ne21;

    const uint32_t qx_sz = use_src0 ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne02 * ne03 : 0;
    const uint32_t qy_sz = use_src1 ? ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne12 * ne13 : 0;
    const uint32_t x_sz = use_src0 ? ggml_vk_align_size(sizeof(ggml_fp16_t) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne02 * ne03 : 0;
    const uint32_t y_sz = use_src1 ? ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne12 * ne13 : 0;
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne22 * ne23;

    switch (node->op) {
    case GGML_OP_REPEAT:
        node->extra = ggml_vk_preallocate_buffers(d_sz, qx_sz, 0, 0, 0);
        break;
    case GGML_OP_GET_ROWS:
        node->extra = ggml_vk_preallocate_buffers(0, 0, 0, 0, 0);
        break;
    case GGML_OP_ADD:
    case GGML_OP_SCALE:
    case GGML_OP_MUL:
    case GGML_OP_RMS_NORM:
        node->extra = ggml_vk_preallocate_buffers(d_sz, transfer_src0 ? qx_sz : 0, transfer_src1 ? qy_sz : 0, 0, 0);
        break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(node->src[0], node->src[1], node)) {
            return;
        }

        node->extra = ggml_vk_preallocate_buffers(d_sz, transfer_src0 ? qx_sz : 0, transfer_src1 ? qy_sz : 0, qx_needs_dequant ? x_sz : 0, qy_needs_dequant ? y_sz : 0);
        break;
    default:
        return;
    }

    // Reuse GPU buffer if previous op is also on GPU
    if (src0_gpu) {
        src0->backend = GGML_BACKEND_GPU;
        ggml_vk_tensor_extra_gpu * src0_extra = (ggml_vk_tensor_extra_gpu *) src0->extra;

        // Replace with data GPU tensor
        src0->data = malloc(sizeof(vk_buffer));
        ggml_vk_pool_malloc(src0_extra->tensor_size, (vk_buffer *)src0->data, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Handle buffer offset alignment issues in 2nd and 3rd dimensions early by changing stride
        ggml_vk_realign_tensor(src0);
    }
    if (src1_gpu) {
        src1->backend = GGML_BACKEND_GPU;
        ggml_vk_tensor_extra_gpu * src1_extra = (ggml_vk_tensor_extra_gpu *) src1->extra;

        // Replace with data GPU tensor
        src1->data = malloc(sizeof(vk_buffer));
        ggml_vk_pool_malloc(src1_extra->tensor_size, (vk_buffer *)src1->data, vk::MemoryPropertyFlagBits::eDeviceLocal);

        ggml_vk_realign_tensor(src1);
    }
}

void ggml_vk_preallocate_buffers() {
    if (vk_prealloc_d.size < vk_prealloc_size_d) {
        // Resize buffer
        if (vk_prealloc_d.size > 0) {
            ggml_vk_destroy_buffer(vk_prealloc_d);
        }
        vk_prealloc_d = ggml_vk_create_buffer(vk_prealloc_size_d, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    if (vk_prealloc_qx.size < vk_prealloc_size_qx) {
        // Resize buffer
        if (vk_prealloc_qx.size > 0) {
            ggml_vk_destroy_buffer(vk_prealloc_qx);
        }
        vk_prealloc_qx = ggml_vk_create_buffer(vk_prealloc_size_qx, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    if (vk_prealloc_qy.size < vk_prealloc_size_qy) {
        // Resize buffer
        if (vk_prealloc_qy.size > 0) {
            ggml_vk_destroy_buffer(vk_prealloc_qy);
        }
        vk_prealloc_qy = ggml_vk_create_buffer(vk_prealloc_size_qy, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    if (vk_prealloc_x.size < vk_prealloc_size_x) {
        // Resize buffer
        if (vk_prealloc_x.size > 0) {
            ggml_vk_destroy_buffer(vk_prealloc_x);
        }
        vk_prealloc_x = ggml_vk_create_buffer(vk_prealloc_size_x, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    if (vk_prealloc_y.size < vk_prealloc_size_y) {
        // Resize buffer
        if (vk_prealloc_y.size > 0) {
            ggml_vk_destroy_buffer(vk_prealloc_y);
        }
        vk_prealloc_y = ggml_vk_create_buffer(vk_prealloc_size_y, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
}

void ggml_vk_build_graph(ggml_tensor * node){
    const bool any_on_device = node->backend == GGML_BACKEND_GPU
        || (node->src[0] != nullptr && (node->src[0]->backend == GGML_BACKEND_GPU || node->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (node->src[1] != nullptr && node->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && node->op != GGML_OP_MUL_MAT) {
        return;
    }

#ifdef VK_DEBUG
    std::cerr << "ggml_vk_build_graph(" << node << ")" << std::endl;
#endif

    vk_semaphore_idx = 0;

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(node->src[0], node->src[1], node);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(node->src[0], node->src[1], node);

        break;
    case GGML_OP_ADD:
        ggml_vk_add(node->src[0], node->src[1], node);

        break;
    case GGML_OP_MUL:
        ggml_vk_mul(node->src[0], node->src[1], node);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(node->src[0], node->src[1], node);

        break;
    case GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(node->src[0], node);

        break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(node->src[0], node->src[1], node)) {
            return;
        }

        ggml_vk_mul_mat(node->src[0], node->src[1], node);

        break;
    default:
        if (any_on_device) {
            std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(node->op) << std::endl;
            GGML_ASSERT(false);
        }
        return;
    }
}

bool ggml_vk_compute_forward(ggml_compute_params * params, ggml_tensor * tensor){
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    if (!any_on_device && tensor->op != GGML_OP_MUL_MAT) {
        return false;
    }

    ggml_vk_tensor_extra_gpu * extra = nullptr;

    switch (tensor->op) {
    case GGML_OP_ADD:
    case GGML_OP_GET_ROWS:
    case GGML_OP_MUL:
    case GGML_OP_SCALE:
    case GGML_OP_RMS_NORM:
        extra = (ggml_vk_tensor_extra_gpu *) tensor->extra;

        break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
            return false;
        }

        extra = (ggml_vk_tensor_extra_gpu *) tensor->extra;

        break;
    default:
        return false;
    }

    if (extra == nullptr) {
        // Graph hasn't been prepared, fall back to CPU
        return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }

#ifdef VK_DEBUG
    std::cerr << "ggml_vk_compute_forward(" << params << ", " << tensor << ")" << std::endl;
#endif

#ifdef GGML_VULKAN_CHECK_RESULTS
    ggml_vk_check_results_0(params, tensor);
#endif

    // Do staging buffer copies
    for (auto& cpy : extra->memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
    ggml_vk_submit(vk_device.transfer_queues[0], extra->in0_seqs, VK_NULL_HANDLE);
    ggml_vk_submit(vk_device.transfer_queues[1], extra->in1_seqs, VK_NULL_HANDLE);
    if (extra->out_seqs.empty()) {
        ggml_vk_submit(vk_device.compute_queue, extra->comp_seqs, vk_fence);
    } else {
        ggml_vk_submit(vk_device.compute_queue, extra->comp_seqs, VK_NULL_HANDLE);
        ggml_vk_submit(vk_device.transfer_queues[1], extra->out_seqs, vk_fence);
    }

    vk::resultCheck(vk_device.device.waitForFences({ vk_fence }, true, uint64_t(-1)), "ggml_vk_compute_forward waitForFences");
    vk_device.device.resetFences({ vk_fence });

    return true;
}

void ggml_vk_graph_cleanup() {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_graph_cleanup()" << std::endl;
#endif
    for (auto * pipeline : vk_gc.pipelines) {
        ggml_vk_pipeline_cleanup(*pipeline);
    }
    vk_gc.pipelines.clear();

    ggml_vk_queue_cleanup(vk_device.compute_queue);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[1]);

    for (size_t i = 0; i < vk_gc.semaphores.size(); i++) {
        vk_device.device.destroySemaphore({ vk_gc.semaphores[i].s });
    }
    vk_gc.semaphores.clear();

    for (size_t i = 0; i < vk_gc.tl_semaphores.size(); i++) {
        vk_device.device.destroySemaphore({ vk_gc.tl_semaphores[i].s });
    }
    vk_gc.tl_semaphores.clear();

    for (auto * extra : vk_gc.extras) {
        if (extra->gpu_buffer != nullptr) {
            ggml_vk_pool_free(*extra->gpu_buffer);
        }
        delete extra;
    }
    vk_gc.extras.clear();
}

#ifdef GGML_VULKAN_CHECK_RESULTS
void ggml_vk_print_graph_origin(const ggml_tensor * tensor, std::vector<const ggml_tensor *>& done, int level = 0) {
    if (std::find(done.begin(), done.end(), tensor) != done.end()) {
        return;
    }
    for (int j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " " << tensor->backend << std::endl;

    done.push_back(tensor);

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

void ggml_vk_check_tensor(const std::string& name, const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        return;
    }
    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    const float val = *(float *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                    if (std::isnan(val)) {
                        std::cerr << "ERROR: TENSOR CHECK " << name << ": Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " val=" << val << std::endl;
                        std::cerr << "tensor->backend: " << tensor->backend << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
                        std::cerr << std::endl;
                        std::vector<const ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ASSERT(false);
                    }
                }
            }
        }
    }
}

void ggml_vk_print_tensor_area(const ggml_tensor * tensor, const void * data, int i0, int i1, int i2, int i3) {
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1]) {
                float val = *(float *) ((char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

size_t ggml_vk_tensor_size(const ggml_tensor * tensor) {
    return std::max(tensor->ne[3]*tensor->nb[3], tensor->nb[1] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
}

void * comp_result;
size_t comp_nb[GGML_MAX_DIMS];
void ggml_vk_check_results_0(ggml_compute_params * params, ggml_tensor * tensor) {
    if (params->ith != 0) {
        return;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    struct ggml_init_params iparams = {
        .mem_size   = 512*1024*1024,
        .mem_buffer = NULL,
    };

    // memory allocation happens here
    struct ggml_context * ctx = ggml_init(iparams);

    struct ggml_tensor * src0_clone = nullptr;
    struct ggml_tensor * src1_clone = nullptr;
    struct ggml_tensor * tensor_clone = nullptr;

    size_t src0_size;
    size_t src1_size;

    if (src0 != nullptr) {
        src0_clone = ggml_dup_tensor(ctx, src0);

        src0_size = ggml_vk_tensor_size(src0);

        src0_clone->data = malloc(src0_size);
        if (src0->backend == GGML_BACKEND_CPU) {
            memcpy(src0_clone->data, src0->data, src0_size);
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (src0->backend == GGML_BACKEND_GPU) {
            if (!ggml_is_contiguous(src0) && ggml_vk_dim01_contiguous(src0)) {
                for (int i3 = 0; i3 < src0->ne[3]; i3++) {
                    for (int i2 = 0; i2 < src0->ne[2]; i2++) {
                        const int idx = i3*src0->ne[2] + i2;
                        ggml_vk_buffer_read((vk_buffer *)src0->data, (idx) * src0->nb[2], ((char *)src0_clone->data + idx * src0_clone->nb[2]), src0->ne[1] * src0->nb[1], vk_device.transfer_queues[0]);
                    }
                }

                src0_clone->nb[0] = src0->nb[0];
                src0_clone->nb[1] = src0->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    src0_clone->nb[i] = src0_clone->nb[i - 1]*src0_clone->ne[i - 1];
                }
            } else {
                ggml_vk_buffer_read((vk_buffer *)src0->data, 0, src0_clone->data, src0_size, vk_device.transfer_queues[0]);
                memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ASSERT(false);
        }

        ggml_vk_check_tensor(std::string(ggml_op_name(tensor->op)) + "->src0", src0_clone);
    }
    if (src1 != nullptr) {
        src1_clone = ggml_dup_tensor(ctx, src1);

        src1_size = ggml_vk_tensor_size(src1);

        src1_clone->data = malloc(src1_size);
        if (src1->backend == GGML_BACKEND_CPU) {
            memcpy(src1_clone->data, src1->data, src1_size);
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (src1->backend == GGML_BACKEND_GPU) {
            if (!ggml_is_contiguous(src1) && ggml_vk_dim01_contiguous(src1)) {
                for (int i3 = 0; i3 < src1->ne[3]; i3++) {
                    for (int i2 = 0; i2 < src1->ne[2]; i2++) {
                        const int idx = i3*src1->ne[2] + i2;
                        ggml_vk_buffer_read((vk_buffer *)src1->data, (idx) * src1->nb[2], ((char *)src1_clone->data + idx * src1_clone->nb[2]), src1->ne[1] * src1->nb[1], vk_device.transfer_queues[0]);
                    }
                }

                src1_clone->nb[0] = src1->nb[0];
                src1_clone->nb[1] = src1->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    src1_clone->nb[i] = src1_clone->nb[i - 1]*src1_clone->ne[i - 1];
                }
            } else {
                ggml_vk_buffer_read((vk_buffer *)src1->data, 0, src1_clone->data, src1_size, vk_device.transfer_queues[0]);
                memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ASSERT(false);
        }

        ggml_vk_check_tensor(std::string(ggml_op_name(tensor->op)) + "->src1", src1_clone);
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        tensor_clone = ggml_mul_mat(ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_MUL) {
        tensor_clone = ggml_mul(ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_SCALE) {
        tensor_clone = ggml_scale(ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ADD) {
        tensor_clone = ggml_add(ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_RMS_NORM) {
        tensor_clone = ggml_rms_norm(ctx, src0_clone, *(float *)tensor->op_params);
    } else {
        std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
        GGML_ASSERT(false);
    }

    struct ggml_cgraph cgraph = ggml_build_forward(tensor_clone);

    ggml_graph_compute_with_ctx(ctx, &cgraph, 8);

    ggml_vk_check_tensor(ggml_op_name(tensor->op), tensor_clone);

    size_t tensor_size = ggml_vk_tensor_size(tensor);

    comp_result = malloc(tensor_size);
    memcpy(comp_result, tensor_clone->data, tensor_size);
    memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * GGML_MAX_DIMS);

    if (src0 != nullptr) {
        free(src0_clone->data);
    }
    if (src1 != nullptr) {
        free(src1_clone->data);
    }

    ggml_free(ctx);
}

void ggml_vk_check_results_1(ggml_compute_params * params, ggml_tensor * tensor) {
    if (params->ith != 0) {
        return;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return;
    }

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    void * tensor_data = tensor->data;

    if (tensor->backend == GGML_BACKEND_GPU) {
        const size_t tensor_size = ggml_vk_tensor_size(tensor);
        tensor_data = malloc(tensor_size);

        ggml_vk_buffer_read((vk_buffer *)tensor->data, 0, tensor_data, tensor_size, vk_device.transfer_queues[0]);
    }

    float first_error_result = -1.0f;
    float first_error_correct = -1.0f;
    std::array<int, 4> first_error = { -1, -1, -1, -1 };
    double avg_err = 0.0;
    size_t counter = 0;

    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    if (tensor->type == GGML_TYPE_F32) {
                        float correct = *(float *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                        float result  = *(float *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);

                        if (std::isnan(correct) || std::isnan(result) || std::isnan(avg_err)) {
                            std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                            std::cerr << "tensor=" << tensor << " tensor->backend: " << tensor->backend << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
                            if (src0 != nullptr) {
                                std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " backend=" << src0->backend << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << std::endl;
                            }
                            if (src1 != nullptr) {
                                std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " backend=" << src1->backend << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << std::endl;
                            }
                            std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                            std::cerr << std::endl;
                            ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                            std::cerr << std::endl;
                            std::vector<const ggml_tensor *> done;
                            ggml_vk_print_graph_origin(tensor, done);
                            GGML_ASSERT(false);
                        }
                        if (first_error[0] == -1 && std::fabs(correct - result) > 0.1f) {
                            first_error[0] = i0;
                            first_error[1] = i1;
                            first_error[2] = i2;
                            first_error[3] = i3;
                            first_error_result = result;
                            first_error_correct = correct;
                        }

                        // Special case, value is infinite, avoid NaN result in avg_err
                        if (!std::isinf(correct) || !std::isinf(result) || correct != result) {
                            avg_err += std::fabs(correct - result);
                        }
                    } else {
                        GGML_ASSERT(false);
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    if (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->backend: " << tensor->backend << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " backend=" << src0->backend << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " backend=" << src1->backend << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<const ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ASSERT(false);
    }

    free(comp_result);
    comp_result = nullptr;

    if (tensor->backend == GGML_BACKEND_GPU) {
        free(tensor_data);
    }
}
#endif

#ifdef VK_CHK_KERNEL
void ggml_vk_test_transfer(size_t ne) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_test_transfer(" << ne << ")" << std::endl;
#endif
    // Check transfers are correct
    vk_buffer buffer = ggml_vk_create_buffer(sizeof(float) * ne, vk::MemoryPropertyFlagBits::eDeviceLocal);

    float* x = (float *) malloc(sizeof(float) * ne);
    float* y = (float *) malloc(sizeof(float) * ne);

    for (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_buffer_write(&buffer, 0, x, sizeof(float) * ne, vk_device.transfer_queues[0]);

    vk_device.transfer_queues[0].queue.waitIdle();

    auto end = std::chrono::high_resolution_clock::now();

    double ms_to_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    begin = std::chrono::high_resolution_clock::now();

    ggml_vk_buffer_read(&buffer, 0, y, sizeof(float) * ne, vk_device.transfer_queues[1]);

    end = std::chrono::high_resolution_clock::now();

    double ms_from_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    double avg_err = 0.0;
    for (size_t i = 0; i < ne; i++) {
        avg_err += std::fabs(x[i] - y[i]);
    }

    double kb = ne * sizeof(float) / 1024.0;

    std::cerr << "TEST TRANSFER " << kb << " KB to_gpu " << ms_to_gpu << "ms (" << kb / ms_to_gpu * 1000.0 / 1024.0 << " MB/s) from_gpu " << ms_from_gpu << "ms (" << kb / ms_from_gpu * 1000.0 / 1024.0 << " MB/s) avg_err=" << avg_err / ne << std::endl;

    ggml_vk_destroy_buffer(buffer);

    free(x);
    free(y);
}

void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_test_matmul_f32(" << m << ", " << n << ", " << k << ", " << num_it << ", " << split_k << ", " << shader_size << ")" << std::endl;
#endif
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    std::vector<vk_sequence> seq;

    vk_pipeline * p;
    std::string shname;
    if (shader_size == 0) {
        p = &vk_pipeline_matmul_f32_aligned_s;
        shname = "F32_ALIGNED_S";
    } else if (shader_size == 1) {
        p = &vk_pipeline_matmul_f32_aligned_m;
        shname = "F32_ALIGNED_M";
    } else if (shader_size == 2) {
        p = &vk_pipeline_matmul_f32_aligned_l;
        shname = "F32_ALIGNED_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            p = &vk_pipeline_matmul_f32_s;
            shname = "F32_S";
        } else if (shader_size == 1) {
            p = &vk_pipeline_matmul_f32_m;
            shname = "F32_M";
        } else if (shader_size == 2) {
            p = &vk_pipeline_matmul_f32_l;
            shname = "F32_L";
        }
    }

    ggml_vk_pipeline_allocate_descriptor_sets(*p, num_it);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, num_it);
    }

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(float) * k * m, &d_X, {});
    ggml_vk_pool_malloc(sizeof(float) * k * n, &d_Y, {});
    ggml_vk_pool_malloc(sizeof(float) * d_ne * split_k, &d_D, {});

    float* x = (float *) malloc(sizeof(float) * x_ne);
    float* y = (float *) malloc(sizeof(float) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = rand() / (float)RAND_MAX;
    }

    seq.push_back(ggml_vk_buffer_write_2d_async(&d_X, 0, x, sizeof(float) * k, sizeof(float) * k, m, vk_device.transfer_queues[0], {}, {}));
    seq.push_back(ggml_vk_buffer_write_2d_async(&d_Y, 0, y, sizeof(float) * k, sizeof(float) * k, n, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    // Wait for transfers to finish
    vk_device.transfer_queues[0].queue.waitIdle();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_it; i++) {
        seq.push_back(ggml_vk_matmul(*p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), m, n, k, k, k, m, split_k, vk_device.compute_queue, {}, {}));
    }

    ggml_vk_submit(vk_device.compute_queue, seq, VK_NULL_HANDLE);

    vk_device.compute_queue.queue.waitIdle();

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_device.transfer_queues[0]);

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

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " split_k=" << split_k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 / num_it << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(d_chk);

    ggml_vk_queue_cleanup(vk_device.transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[1]);
    ggml_vk_queue_cleanup(vk_device.compute_queue);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    ggml_vk_pipeline_cleanup(*p);
    ggml_vk_pipeline_cleanup(vk_pipeline_matmul_split_k_reduce);

    free(x);
    free(y);
    free(d);
}

void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_test_matmul_f16(" << m << ", " << n << ", " << k << ", " << num_it << ", " << split_k << ", " << shader_size << ")" << std::endl;
#endif
    if (!vk_device.fp16) {
        return;
    }
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    std::vector<vk_sequence> seq;

    vk_pipeline * p;
    std::string shname;
    if (shader_size == 0) {
        p = &vk_pipeline_matmul_f16_aligned_s;
        shname = "F16_ALIGNED_S";
    } else if (shader_size == 1) {
        p = &vk_pipeline_matmul_f16_aligned_m;
        shname = "F16_ALIGNED_M";
    } else if (shader_size == 2) {
        p = &vk_pipeline_matmul_f16_aligned_l;
        shname = "F16_ALIGNED_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            p = &vk_pipeline_matmul_f16_s;
            shname = "F16_S";
        } else if (shader_size == 1) {
            p = &vk_pipeline_matmul_f16_m;
            shname = "F16_M";
        } else if (shader_size == 2) {
            p = &vk_pipeline_matmul_f16_l;
            shname = "F16_L";
        }
    }

    ggml_vk_pipeline_allocate_descriptor_sets(*p, num_it);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, num_it);
    }

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * k * m, &d_X, {});
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * k * n, &d_Y, {});
    ggml_vk_pool_malloc(sizeof(float) * d_ne * split_k, &d_D, {});

    ggml_fp16_t* x = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * x_ne);
    ggml_fp16_t* y = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }

    seq.push_back(ggml_vk_buffer_write_2d_async(&d_X, 0, x, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, m, vk_device.transfer_queues[0], {}, {}));
    seq.push_back(ggml_vk_buffer_write_2d_async(&d_Y, 0, y, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, n, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    // Wait for transfers to finish
    vk_device.transfer_queues[0].queue.waitIdle();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_it; i++) {
        seq.push_back(ggml_vk_matmul(*p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), m, n, k, k, k, m, split_k, vk_device.compute_queue, {}, {}));
    }

    ggml_vk_submit(vk_device.compute_queue, seq, VK_NULL_HANDLE);

    vk_device.compute_queue.queue.waitIdle();

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_device.transfer_queues[0]);

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

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " split_k=" << split_k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 / num_it << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(fx);
    free(fy);
    free(d_chk);

    ggml_vk_queue_cleanup(vk_device.transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[1]);
    ggml_vk_queue_cleanup(vk_device.compute_queue);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    ggml_vk_pipeline_cleanup(*p);
    ggml_vk_pipeline_cleanup(vk_pipeline_matmul_split_k_reduce);

    free(x);
    free(y);
    free(d);
}

void ggml_vk_test_matmul_f16_f32(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_test_matmul_f16(" << m << ", " << n << ", " << k << ", " << num_it << ", " << split_k << ", " << shader_size << ")" << std::endl;
#endif
    if (!vk_device.fp16) {
        return;
    }
    const size_t x_ne = m * k;
    const size_t y_ne = k * n;
    const size_t d_ne = m * n;

    std::vector<vk_sequence> seq;

    vk_pipeline * p;
    std::string shname;
    if (shader_size == 0) {
        p = &vk_pipeline_matmul_f16_f32_aligned_s;
        shname = "F16_F32_ALIGNED_S";
    } else if (shader_size == 1) {
        p = &vk_pipeline_matmul_f16_f32_aligned_m;
        shname = "F16_F32_ALIGNED_M";
    } else if (shader_size == 2) {
        p = &vk_pipeline_matmul_f16_f32_aligned_l;
        shname = "F16_F32_ALIGNED_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            p = &vk_pipeline_matmul_f16_f32_s;
            shname = "F16_F32_S";
        } else if (shader_size == 1) {
            p = &vk_pipeline_matmul_f16_f32_m;
            shname = "F16_F32_M";
        } else if (shader_size == 2) {
            p = &vk_pipeline_matmul_f16_f32_l;
            shname = "F16_F32_L";
        }
    }

    ggml_vk_pipeline_allocate_descriptor_sets(*p, num_it);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, num_it);
    }

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * k * m, &d_X, {});
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * k * n, &d_Y, {});
    ggml_vk_pool_malloc(sizeof(float) * d_ne * split_k, &d_D, {});

    ggml_fp16_t* x = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * x_ne);
    float* y = (float *) malloc(sizeof(float) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < y_ne; i++) {
        y[i] = rand() / (float)RAND_MAX;
    }

    seq.push_back(ggml_vk_buffer_write_2d_async(&d_X, 0, x, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, m, vk_device.transfer_queues[0], {}, {}));
    seq.push_back(ggml_vk_buffer_write_2d_async(&d_Y, 0, y, sizeof(float) * k, sizeof(float) * k, n, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    // Wait for transfers to finish
    vk_device.transfer_queues[0].queue.waitIdle();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_it; i++) {
        seq.push_back(ggml_vk_matmul(*p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), m, n, k, k, k, m, split_k, vk_device.compute_queue, {}, {}));
    }

    ggml_vk_submit(vk_device.compute_queue, seq, VK_NULL_HANDLE);

    vk_device.compute_queue.queue.waitIdle();

    auto end = std::chrono::high_resolution_clock::now();

    // copy dst to host
    ggml_vk_buffer_read(&d_D, 0, d, sizeof(float) * d_ne, vk_device.transfer_queues[0]);

    float * fx = (float *) malloc(sizeof(float) * x_ne);
    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    ggml_fp16_to_fp32_row(x, fx, x_ne);

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            m, n, k,
            1.0f,    fx, k,
                     y, k,
            0.0f,    d_chk, m);

    double avg_err = 0.0;

    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            avg_err += std::fabs(d[c * m + r] - d_chk[c * m + r]);
        }
    }

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " split_k=" << split_k << " matmul " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0 / num_it << "ms avg_err=" << avg_err / (m * n) << std::endl;

    free(fx);
    free(d_chk);

    ggml_vk_queue_cleanup(vk_device.transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[1]);
    ggml_vk_queue_cleanup(vk_device.compute_queue);

    ggml_vk_pool_free(d_X);
    ggml_vk_pool_free(d_Y);
    ggml_vk_pool_free(d_D);

    ggml_vk_pipeline_cleanup(*p);
    ggml_vk_pipeline_cleanup(vk_pipeline_matmul_split_k_reduce);

    free(x);
    free(y);
    free(d);
}
#endif
