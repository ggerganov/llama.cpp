#include "ggml-vulkan.h"

#ifdef VK_CHK_KERNEL
#include <cblas.h>
#include <chrono>
#endif

#ifdef VK_PROFILE
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

#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <vector>
#include <mutex>
#include <sstream>

#include <shaderc/shaderc.hpp>

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
    vk_queue() {};
    vk_queue(const vk_queue& b) : queue_family_index(b.queue_family_index), queue(b.queue), pool(b.pool), cmd_buffer_idx(b.cmd_buffer_idx), cmd_buffers(b.cmd_buffers), semaphore_idx(b.semaphore_idx), semaphores(b.semaphores), stage_flags(b.stage_flags) {}

    vk_queue& operator=(const vk_queue& b) {
        if (this != &b) {
            queue_family_index = b.queue_family_index;
            queue = b.queue;
            pool = b.pool;
            cmd_buffer_idx = b.cmd_buffer_idx;
            cmd_buffers = b.cmd_buffers;
            semaphore_idx = b.semaphore_idx;
            semaphores = b.semaphores;
            stage_flags = b.stage_flags;
        }
        return *this;
    }

    uint32_t queue_family_index;
    vk::Queue queue;
    vk::CommandPool pool;
    uint32_t cmd_buffer_idx;
    std::vector<vk::CommandBuffer> cmd_buffers;
    uint32_t semaphore_idx;
    std::vector<vk::Semaphore> semaphores;

    vk::PipelineStageFlags stage_flags;

    std::mutex mutex;
};

struct vk_submission {
    vk::CommandBuffer buffer;
    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<vk::Semaphore> signal_semaphores;
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
    int stride_x;
    int stride_y;
    int stride_d;
    int x_offset;
    int y_offset;
    int d_offset;
    float scale;
};

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct ggml_vk_tensor_extra_gpu {
    uint32_t batch_size;
    std::vector<uint32_t> buffer_idx;

    std::vector<vk_staging_memcpy> memcpys;
    std::vector<vk_sequence> in0_seqs;
    std::vector<vk_sequence> in1_seqs;
    std::vector<vk_sequence> comp_seqs;
    std::vector<vk_sequence> out_seqs;
};

struct ggml_vk_garbage_collector {
    std::vector<vk_pipeline *> pipelines;
    std::vector<ggml_vk_tensor_extra_gpu *> extras;
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
vk_pipeline vk_pipeline_mul_f32;
vk_pipeline vk_pipeline_add_f32, vk_pipeline_add_f16_f32_f16;
vk_pipeline vk_pipeline_scale_f32;

static ggml_vk_garbage_collector vk_gc;
static std::vector<std::tuple<void*, size_t, vk_buffer>> vk_pinned_memory;
static std::vector<size_t> vk_preallocated_buffer_sizes;
static std::vector<vk_buffer> vk_preallocated_buffers;
static vk::Fence vk_fence;

static std::vector<uint32_t> ggml_vk_compile_shader(const std::string& name, const std::string& src, std::vector<std::string>&& defines) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_compile_shader(" << name << ", " << src << ")" << std::endl;
#endif
    GGML_ASSERT(defines.size() % 2 == 0);

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    for (size_t i = 0; i < defines.size(); i += 2) {
        options.AddMacroDefinition(defines[i], defines[i + 1]);
    }

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(src, shaderc_compute_shader, name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        shaderc::PreprocessedSourceCompilationResult prep_res = compiler.PreprocessGlsl(src, shaderc_compute_shader, name.c_str(), options);

        std::string prep_src = std::string{ prep_res.begin(), prep_res.end() };

        std::stringstream ss(prep_src);
        std::string line;
        int counter = 1;
        while(std::getline(ss, line, '\n')){
            std::cout << std::setw(3) << counter++ << std::setw(1) << ": " << line << std::endl;
        }
        std::cerr << "ggml_vulkan: Shader compile error in " << name << ": " << module.GetErrorMessage();
        GGML_ASSERT(false);
    }

    return {module.cbegin(), module.cend()};
}

static vk_pipeline ggml_vk_create_pipeline(const std::string& name, size_t spv_size, const uint32_t* spv_data, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, std::vector<int>&& specialization_constants, uint32_t align) {
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

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, spv_data);
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

static vk_pipeline ggml_vk_create_pipeline_from_string(const std::string& name, const std::string& src, std::vector<std::string>&& defines, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, std::vector<int>&& specialization_constants, uint32_t align) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_pipeline_from_string(" << name << ", " << entrypoint << ", " << parameter_count << ", " << push_constant_size << ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " << align << ")" << std::endl;
#endif

    const std::vector<uint32_t> spv = ggml_vk_compile_shader(name, src, std::move(defines));
    return ggml_vk_create_pipeline(name, spv.size() * sizeof(uint32_t), spv.data(), entrypoint, parameter_count, push_constant_size, wg_denoms, std::move(specialization_constants), align);
}

static vk_pipeline ggml_vk_create_pipeline_from_file(const std::string& path, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, std::vector<int>&& specialization_constants, uint32_t align) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_pipeline_from_file(" << path << ", " << entrypoint << ", " << parameter_count << ", " << push_constant_size << ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " << align << ")" << std::endl;
#endif

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

    return ggml_vk_create_pipeline(path, matmul_shader_contents.size(), reinterpret_cast<uint32_t *>(matmul_shader_contents.data()), entrypoint, parameter_count, push_constant_size, wg_denoms, std::move(specialization_constants), align);
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

static vk_submission ggml_vk_create_submission(vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_submission()" << std::endl;
#endif
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(q);
    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
    return s;
}

static vk_sequence ggml_vk_create_sequence_1(vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores) {
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

    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    for (const auto& sequence : sequences) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(q.stage_flags);
            }
            submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                submission.wait_semaphores.data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer,
                (uint32_t) submission.signal_semaphores.size(),
                submission.signal_semaphores.data()
            });
        }
    }

    std::lock_guard<std::mutex> guard(q.mutex);

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
    q.semaphore_idx = 0;

    q.queue = vk_device.device.getQueue(queue_family_index, queue_index);

    q.stage_flags = stage_flags;

    return q;
}

static vk::Semaphore ggml_vk_create_semaphore(vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_create_semaphore()" << std::endl;
#endif
    if (q.semaphores.size() > q.semaphore_idx) {
        // Reuse semaphore
        return q.semaphores[q.semaphore_idx++];
    }

    vk::Semaphore semaphore = vk_device.device.createSemaphore({});
    q.semaphores.push_back(semaphore);
    q.semaphore_idx++;

    return semaphore;
}

static void ggml_vk_queue_cleanup(vk_queue& q) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_queue_cleanup()" << std::endl;
#endif
    // Requires semaphores and command buffers to be done

    q.semaphore_idx = 0;

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

static inline vk_subbuffer ggml_vk_subbuffer(vk_buffer& buf) {
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

static inline bool ggml_vk_build_shader_type_defines(std::stringstream& stream, ggml_type type, bool compat) {
    switch(type) {
    case GGML_TYPE_F16:
        stream << shader_f16_defines << (compat ? shader_f16_dequant_func_compat : shader_f16_dequant_func);
        return true;
    case GGML_TYPE_Q4_0:
        stream << shader_q4_0_defines << (compat ? shader_q4_0_dequant_func_compat : shader_q4_0_dequant_func);
        return true;
    case GGML_TYPE_Q4_1:
        stream << shader_q4_1_defines << (compat ? shader_q4_1_dequant_func_compat : shader_q4_1_dequant_func);
        return true;
    case GGML_TYPE_Q5_0:
        stream << shader_q5_0_defines << (compat ? shader_q5_0_dequant_func_compat : shader_q5_0_dequant_func);
        return true;
    case GGML_TYPE_Q5_1:
        stream << shader_q5_1_defines << (compat ? shader_q5_1_dequant_func_compat : shader_q5_1_dequant_func);
        return true;
    case GGML_TYPE_Q8_0:
        stream << shader_q8_0_defines << (compat ? shader_q8_0_dequant_func_compat : shader_q8_0_dequant_func);
        return true;
    case GGML_TYPE_Q6_K:
        stream << shader_q6_K_defines;
        return true;
    default:
        return false;
    }
}

static void ggml_vk_generate_shaders() {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_generate_shaders()" << std::endl;
#endif
    std::cerr << "ggml_vulkan: Generating and compiling shaders to SPIR-V" << std::endl;

    // mulmat
    auto warptile_l = { 128, 128, 128, 16, 64, 64, 2, 4, 4 };
    auto warptile_m = { 128,  64,  64, 16, 32, 32, 2, 4, 2 };
    auto warptile_s = {  32,  32,  32,  8, 32, 32, 2, 2, 2 };

    std::string shader_float_type;
    std::string load_vec;
    std::string vec_type_f16;
    std::string vec_type;
    if (vk_device.fp16) {
        shader_float_type = shader_f16;
        load_vec = "8";
        vec_type_f16 = "f16mat2x4";
        vec_type = "mat2x4";
    } else {
        shader_float_type = shader_f32;
        load_vec = "4";
        vec_type_f16 = "f16vec4";
        vec_type = "vec4";
    }

    std::stringstream stream;
    stream << mulmat_head << shader_float_type << mulmat_body;
    vk_pipeline_matmul_f32_l = ggml_vk_create_pipeline_from_string("matmul_f32_l", stream.str(), { "A_TYPE", "float", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f32_m = ggml_vk_create_pipeline_from_string("matmul_f32_m", stream.str(), { "A_TYPE", "float", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f32_s = ggml_vk_create_pipeline_from_string("matmul_f32_s", stream.str(), { "A_TYPE", "float", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);
    vk_pipeline_matmul_f32_aligned_l = ggml_vk_create_pipeline_from_string("matmul_f32_aligned_l", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f32_aligned_m = ggml_vk_create_pipeline_from_string("matmul_f32_aligned_m", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f32_aligned_s = ggml_vk_create_pipeline_from_string("matmul_f32_aligned_s", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

    stream.str("");
    stream.clear();
    stream << mulmat_head << shader_float_type << mulmat_body;
    vk_pipeline_matmul_f16_l = ggml_vk_create_pipeline_from_string("matmul_f16_l", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float16_t", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f16_m = ggml_vk_create_pipeline_from_string("matmul_f16_m", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float16_t", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f16_s = ggml_vk_create_pipeline_from_string("matmul_f16_s", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float16_t", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

    vk_pipeline_matmul_f16_aligned_l = ggml_vk_create_pipeline_from_string("matmul_f16_aligned_l", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type_f16, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f16_aligned_m = ggml_vk_create_pipeline_from_string("matmul_f16_aligned_m", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type_f16, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f16_aligned_s = ggml_vk_create_pipeline_from_string("matmul_f16_aligned_s", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type_f16, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

    vk_pipeline_matmul_f16_f32_l = ggml_vk_create_pipeline_from_string("matmul_f16_f32_l", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f16_f32_m = ggml_vk_create_pipeline_from_string("matmul_f16_f32_m", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f16_f32_s = ggml_vk_create_pipeline_from_string("matmul_f16_f32_s", stream.str(), { "A_TYPE", "float16_t", "B_TYPE", "float", "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);
    vk_pipeline_matmul_f16_f32_aligned_l = ggml_vk_create_pipeline_from_string("matmul_f16_f32_aligned_l", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), {128, 128, 1}, warptile_l, 128);
    vk_pipeline_matmul_f16_f32_aligned_m = ggml_vk_create_pipeline_from_string("matmul_f16_f32_aligned_m", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 64,  64, 1}, warptile_m, 64);
    vk_pipeline_matmul_f16_f32_aligned_s = ggml_vk_create_pipeline_from_string("matmul_f16_f32_aligned_s", stream.str(), { "LOAD_VEC", load_vec, "A_TYPE", vec_type_f16, "B_TYPE", vec_type, "D_TYPE", "float" }, "main", 3, 7 * sizeof(int), { 32,  32, 1}, warptile_s, 32);

    // Build dequant shaders
    vk_pipeline_dequant[GGML_TYPE_F32] = ggml_vk_create_pipeline_from_string("f32_to_f16", f32_to_f16_src, {}, "main", 2, 4 * sizeof(int), {64, 1, 1}, {}, 1);

    for (int i = 0; i < VK_NUM_TYPES; i++) {
        stream.str("");
        stream.clear();

        stream << dequant_head << shader_int8_ext << shader_float_type;

        if (!ggml_vk_build_shader_type_defines(stream, (ggml_type)i, !vk_device.fp16)) {
            continue;
        }

        switch ((ggml_type)i) {
        case GGML_TYPE_Q6_K:
            stream << dequant_q6_K_body;
            break;
        default:
            stream << dequant_body;
            break;
        }

        vk_pipeline_dequant[i] = ggml_vk_create_pipeline_from_string("dequant_" + std::string(ggml_type_name((ggml_type)i)), stream.str(), { "D_TYPE", "float16_t" }, "main", 2, 4 * sizeof(int), {256 * 32, 1, 1}, {}, 1);
    }

    // mul mat vec
    for (int i = 0; i < VK_NUM_TYPES; i++) {
        stream.str("");
        stream.clear();

        stream << mul_mat_vec_head << shader_int8_ext << shader_float_type;

        if (!ggml_vk_build_shader_type_defines(stream, (ggml_type)i, !vk_device.fp16)) {
            continue;
        }

        switch ((ggml_type)i) {
        case GGML_TYPE_Q6_K:
            stream << mul_mat_vec_q6_K_body;
            break;
        default:
            stream << mul_mat_vec_body;
            break;
        }

        vk_pipeline_dequant_mul_mat_vec[i] = ggml_vk_create_pipeline_from_string("mul_mat_vec_" + std::string(ggml_type_name((ggml_type)i)), stream.str(), { "B_TYPE", "float", "D_TYPE", "float16_t", "K_QUANTS_PER_ITERATION", std::to_string(K_QUANTS_PER_ITERATION) }, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
        vk_pipeline_dequant_mul_mat_vec_f32[i] = ggml_vk_create_pipeline_from_string("mul_mat_vec_" + std::string(ggml_type_name((ggml_type)i)) + "_f32", stream.str(), { "B_TYPE", "float", "D_TYPE", "float", "K_QUANTS_PER_ITERATION", std::to_string(K_QUANTS_PER_ITERATION) }, "main", 3, 1 * sizeof(int), {1, 1, 1}, {}, 1);
    }

    // add
    stream.str("");
    stream.clear();

    stream << add_head << shader_float_type << add_body;
    vk_pipeline_add_f32 = ggml_vk_create_pipeline_from_string("add_f32", stream.str(), { "X_TYPE", "float", "Y_TYPE", "float", "D_TYPE", "float" }, "main", 3, sizeof(vk_op_push_constants), {32, 32, 1}, {}, 1);
    stream.str("");
    stream.clear();

    stream << add_head << shader_float_type << add_body;
    vk_pipeline_add_f16_f32_f16 = ggml_vk_create_pipeline_from_string("add_f16_f32_f16", stream.str(), { "X_TYPE", "float16_t", "Y_TYPE", "float", "D_TYPE", "float16_t" }, "main", 3, sizeof(vk_op_push_constants), {32, 32, 1}, {}, 1);

    // Static shaders
    vk_pipeline_matmul_split_k_reduce = ggml_vk_create_pipeline_from_string("split_k_reduce", mulmat_split_k_reduce_src, {}, "main", 1, 3 * sizeof(int), {32, 32, 1}, {}, 1);
    vk_pipeline_mul_f32 = ggml_vk_create_pipeline_from_string("mul_f32", mul_f32_src, { "X_TYPE", "float", "Y_TYPE", "float", "D_TYPE", "float" }, "main", 3, sizeof(vk_op_push_constants), {32, 32, 1}, {}, 1);

    vk_pipeline_scale_f32 = ggml_vk_create_pipeline_from_string("scale_f32", scale_src, { "X_TYPE", "float", "D_TYPE", "float" }, "main", 3, sizeof(vk_op_push_constants), {32, 32, 1}, {}, 1);
}

void ggml_vk_test_transfer(size_t ne);
void ggml_vk_test_matmul_f32(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size);
void ggml_vk_test_matmul_f16(size_t m, size_t n, size_t k, size_t num_it, int split_k, int shader_size);
void ggml_vk_test_buffer_write_zeropad(size_t m, size_t k, size_t align);

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
    const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eVideoDecodeKHR | vk::QueueFlagBits::eProtected | vk::QueueFlagBits::eOpticalFlowNV, compute_queue_family_index, 2);

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

    std::cerr << "Queue Families:" << std::endl;
    for(size_t i = 0; i < queue_family_props.size(); i++) {
        std::cerr << i << ": Queues: "  + std::to_string(queue_family_props[i].queueCount) << " flags: " + to_string(queue_family_props[i].queueFlags) << std::endl;
    }

    std::cerr << "Using compute queue family " << compute_queue_family_index << " and transfer queue family " << transfer_queue_family_index << std::endl;

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
    ggml_vk_generate_shaders();

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
    ggml_vk_test_buffer_write_zeropad(233, 97, 128);
    ggml_vk_test_buffer_write_zeropad(233, 97, 1);
    ggml_vk_test_buffer_write_zeropad(256, 128, 1);

    int step = 16;
    for (size_t m = step; m < 64; m += step) {
        ggml_vk_test_transfer(1024 * 1024 * m);
    }
    const std::vector<size_t> vals {
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
    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 0);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 0);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 1);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 1);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 2);
        ggml_vk_test_matmul_f32(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 2);
        std::cerr << std::endl;

        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 0);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 0);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 1);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 1);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 1, 2);
        ggml_vk_test_matmul_f16(vals[i], vals[i + 1], vals[i + 2], 1000, 4, 2);
        std::cerr << std::endl << std::endl;
    }
#endif
}

static inline vk_pipeline* ggml_vk_get_to_fp16(ggml_type type) {
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
        case GGML_TYPE_Q6_K:
            break;
        default:
            return nullptr;
    }

    return &vk_pipeline_dequant[type];
}

static inline vk_pipeline* ggml_vk_get_dequantize_mul_mat_vec(ggml_type type, bool f16_y) {
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
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_host_free()" << std::endl;
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

static void ggml_vk_end_submission(vk_submission& s, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores) {
    s.buffer.end();

    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
}

static vk_sequence ggml_vk_buffer_write_2d_async(vk_buffer* dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
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

static inline size_t ggml_vk_align_size(size_t width, size_t align) {
    return CEIL_DIV(width, align) * align;
}

static vk_sequence ggml_vk_buffer_write_2d_async_zeropad(vk_buffer* dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, size_t align, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores, vk_submission* s = nullptr) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_buffer_write_2d_async_zeropad(" << offset << ", " << spitch << ", " << width << ", " << height << ", " << align << ")" << std::endl;
#endif
    // Outdated
    GGML_ASSERT(false);
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_2d_async_zeropad dst buffer is host_visible. Use synchronous write." << std::endl;
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

    // Align slices to the value of align
    const uint32_t padded_width = ggml_vk_align_size(width, align);

    bool reuse_submission = false;
    vk_submission submission;
    if (s == nullptr) {
        submission = ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores));
        s = &submission;
        reuse_submission = true;
    }

    if (buf != nullptr) {
        std::vector<vk::BufferCopy> slices(1);
        if (width == padded_width && width == spitch) {
            // Only do single write if no padding happens
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * padded_width;
                slices[i].size = width;
            }
        }

        if (reuse_submission) {
            s->buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        }
        ggml_vk_sync_buffers(s->buffer, { ggml_vk_subbuffer(*dst) }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eMemoryWrite, false);
        if (padded_width > width) {
            s->buffer.fillBuffer(dst->buffer, 0, VK_WHOLE_SIZE, 0);
        }
        s->buffer.pipelineBarrier(
            q.stage_flags,
            q.stage_flags,
            {},
            {},
            {
                { vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eMemoryWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, dst->buffer, 0, VK_WHOLE_SIZE }
            },
            {}
        );
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

    vk::BufferCopy buf_copy = {
        0,
        offset,
        padded_width * height};

    if (reuse_submission) {
        s->buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    }
    ggml_vk_sync_buffers(s->buffer, { ggml_vk_subbuffer(*dst) }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite, false);
    s->buffer.copyBuffer(dst->sb_write->buffer, dst->buffer, { buf_copy });
    if (reuse_submission) {
        s->buffer.end();
    }

    const size_t zeropad = padded_width - width;

    if (width == padded_width && width == spitch) {
        memcpy(dst->sb_write->ptr, src, width * height);
    } else {
        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->sb_write->ptr + i * padded_width, (const uint8_t *) src + i * spitch, width);
            memset((uint8_t *)dst->sb_write->ptr + i * padded_width + width, 0, zeropad);
        }
    }

    return { *s };
}

static vk_sequence ggml_vk_buffer_write_async(vk_buffer* dst, size_t offset, const void * src, size_t size, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
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

static vk_sequence ggml_vk_buffer_read_async(vk_buffer* src, size_t offset, void * dst, size_t size, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores, vk_submission* s = nullptr) {
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
        std::lock_guard<std::mutex> guard(q.mutex);
        q.queue.submit({ submit_info }, fence);
        vk::resultCheck(vk_device.device.waitForFences({ fence }, true, uint64_t(-1)), "vk_buffer_read staging waitForFences");
        vk_device.device.destroyFence(fence);
        memcpy(dst, src->sb_read->ptr, size);
    }
}

static vk_sequence ggml_vk_h2d_tensor_2d(vk_buffer* dst, size_t offset, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores, vk_submission* s = nullptr, std::vector<vk_staging_memcpy>* pre_staging = nullptr) {
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
        return ggml_vk_buffer_write_async(dst, offset, x, ne1*nb1, q, std::move(wait_semaphores), std::move(signal_semaphores), s, pre_staging);
    }
    if (nb0 == ts) {
        return ggml_vk_buffer_write_2d_async(dst, offset, x, nb1, row_length, ne1, q, std::move(wait_semaphores), std::move(signal_semaphores), s, pre_staging);
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

static vk_sequence ggml_vk_h2d_tensor_2d_f32_to_f16(vk_buffer* dst, size_t offset, const struct ggml_tensor * src, uint64_t i3, uint64_t i2, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_h2d_tensor_2d()" << std::endl;
#endif
    GGML_ASSERT(src->type == GGML_TYPE_F32);

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

    const uint32_t copy_size = sizeof(ggml_fp16_t) * ne0 * ne1;

    if (dst->sb_write == nullptr) {
        dst->sb_write = new vk_buffer;
        *dst->sb_write = ggml_vk_create_buffer(dst->size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    ggml_fp16_t * tmp = (ggml_fp16_t *) ((uint8_t *) dst->sb_write->ptr + offset);
    const uint8_t * x = (const uint8_t *) src->data + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == row_length) {
        ggml_fp32_to_fp16_row((const float *) x, tmp, ne0*ne1);

        vk_submission s = ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores));

        vk::BufferCopy buf_copy = {
            offset,
            offset,
            copy_size,
        };

        s.buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        ggml_vk_sync_buffers(s.buffer, { { *dst, (uint32_t)offset, copy_size } }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite, false);
        s.buffer.copyBuffer(dst->sb_write->buffer, dst->buffer, { buf_copy });
        s.buffer.end();

        return { s };
    }
    if (nb0 == ts) {
        for (uint64_t i1 = 0; i1 < ne1; i1++) {
            ggml_fp32_to_fp16_row((const float *) (x + i1*nb1), tmp + i1*ne0, ne0);
        }

        vk_submission s = ggml_vk_create_submission(q, std::move(wait_semaphores), std::move(signal_semaphores));

        vk::BufferCopy buf_copy = {
            offset,
            offset,
            copy_size,
        };

        s.buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        ggml_vk_sync_buffers(s.buffer, { { *dst, (uint32_t)offset, copy_size } }, q, vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eTransferWrite, false);
        s.buffer.copyBuffer(dst->sb_write->buffer, dst->buffer, { buf_copy });
        s.buffer.end();

        return { s };
    }
    GGML_ASSERT(false);
}

static int ggml_vk_guess_split_k(int m, int n, int k) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_guess_split_k()";
#endif
    if (k > 128 && (m < 128 || n < 128)) {
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
    std::cerr << "ggml_vk_guess_matmul_pipeline_padding()" << std::endl;
#endif
    if (m <= 32 || n <= 32) {
        return vk_pipeline_matmul_f32_s.align;
    }
    if (m <= 64 || n <= 64) {
        return vk_pipeline_matmul_f32_m.align;
    }
    return vk_pipeline_matmul_f32_l.align;
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

static vk_sequence ggml_vk_matmul(vk_pipeline& pipeline, vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, int m, int n, int k, int stride_a, int stride_b, int stride_d, int split_k, vk_queue& q, std::vector<vk::Semaphore> wait_semaphores, std::vector<vk::Semaphore> signal_semaphores) {
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
    const std::vector<int> pc1 = { m, n, k, stride_a, stride_b, stride_d, CEIL_DIV(stride_a, split_k) };
    ggml_vk_dispatch_pipeline(s, pipeline, { a, b, d }, pc1.size() * sizeof(int), pc1.data(), { (uint32_t)m * split_k, (uint32_t)n, 1 });
    ggml_vk_sync_buffers(s.buffer, { d }, q, vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, true);
    const std::vector<int> pc2 = { m, n, split_k };
    ggml_vk_dispatch_pipeline(s, vk_pipeline_matmul_split_k_reduce, { d }, pc2.size() * sizeof(int), pc2.data(), { (uint32_t)m, (uint32_t)n, 1 });
    ggml_vk_end_submission(s, std::move(wait_semaphores), std::move(signal_semaphores));

    return { s };
}

static void ggml_vk_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_f32((type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3];
    std::cerr << "), (type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3];
    std::cerr << "), (type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << "),)" << std::endl;
#endif
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

    const int split_k = ggml_vk_guess_split_k(ne01, ne11, ne10);

    const int kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ne01, ne11));

    const bool load_x = src0->backend == GGML_BACKEND_GPU;

    vk_pipeline * pipeline = ggml_vk_guess_matmul_pipeline(false, false, ne01, ne11, ne10 == kpad);

    const uint32_t x_sz = ggml_vk_align_size(sizeof(float) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(sizeof(float) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    uint32_t buffer_idx = 0;

    vk_buffer* d_D = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    vk_buffer* d_X;
    vk_buffer* d_Y;
    if (load_x) {
        d_X = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    } else {
        d_X = (vk_buffer *) src0->data;
    }
    d_Y = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];

    // Allocate descriptor sets
    ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne02 * ne03);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, ne02 * ne03);
    }

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint32_t x_offset = load_x ? x_sz * (i03 * ne02 + i02) : 0;
            const uint32_t y_offset = y_sz * (i03 * ne02 + i02);
            const uint32_t d_offset = d_sz * (i03 * ne02 + i02);

            vk::Semaphore s_x;
            vk::Semaphore s_y = ggml_vk_create_semaphore(vk_device.compute_queue);
            std::vector<vk::Semaphore> semaphores = { s_y };
            // copy data to device
            if (load_x) {
                s_x = ggml_vk_create_semaphore(vk_device.compute_queue);
                semaphores.push_back(s_x);
                // Wait for previous matmul to be done before writing to the input buffers again
                extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_X, x_offset, src0, i03, i02, vk_device.transfer_queues[0], {}, { s_x }, nullptr, &extra->memcpys));
            }

            // Wait for previous matmul to be done before writing to the input buffers again
            extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Y, y_offset, src1, i03, i02, vk_device.transfer_queues[1], {}, { s_y }, nullptr, &extra->memcpys));

            // compute
            vk::Semaphore s_mm = ggml_vk_create_semaphore(vk_device.compute_queue);

            extra->comp_seqs.push_back(ggml_vk_matmul(*pipeline, { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz }, ne01, ne11, ne10, ne10, ne10, ne01, split_k, vk_device.compute_queue, std::move(semaphores), { s_mm }));

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, vk_device.transfer_queues[1], { s_mm }, {}));
        }
    }
}

static void ggml_vk_mul_mat_q_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_q_f16((type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3];
    std::cerr << "), (type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3];
    std::cerr << "), (type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << "),)" << std::endl;
#endif
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    vk_queue& compq = vk_device.compute_queue;
    vk_queue& tr0q = vk_device.transfer_queues[0];
    vk_queue& tr1q = vk_device.transfer_queues[1];
    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = src0->type != GGML_TYPE_F16;
    const bool qy_needs_dequant = src1->type != GGML_TYPE_F16 && !f16_f32_kernel;
    const bool dq = qx_needs_dequant || qy_needs_dequant;

    const bool load_x = src0->backend != GGML_BACKEND_GPU;
    const bool load_y = src1->backend != GGML_BACKEND_GPU;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const int split_k = ggml_vk_guess_split_k(ne01, ne11, ne10);

    const int kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ne01, ne11));

    vk_pipeline * pipeline = ggml_vk_guess_matmul_pipeline(true, !f16_f32_kernel, ne01, ne11, ne10 == kpad);

    const uint32_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t qy_sz = ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t x_sz = ggml_vk_align_size(sizeof(ggml_fp16_t) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    uint32_t buffer_idx = 0;

    vk_buffer* d_D = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    GGML_ASSERT(d_D->size >= d_sz);
    vk_buffer* d_Qx;
    vk_buffer* d_Qy;
    vk_buffer* d_X;
    vk_buffer* d_Y;
    if (load_x) {
        d_Qx = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
        GGML_ASSERT(d_Qx->size >= qx_sz);
    } else {
        d_Qx = (vk_buffer *) src0->data;
    }
    if (load_y) {
        d_Qy = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
        GGML_ASSERT(d_Qy->size >= qy_sz);
    } else {
        d_Qy = (vk_buffer *) src1->data;
    }
    if (qx_needs_dequant) {
        d_X = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
        GGML_ASSERT(d_X->size >= x_sz);
    } else {
        d_X = d_Qx;
        GGML_ASSERT(qx_sz == x_sz);  // NOLINT
    }
    if (qy_needs_dequant) {
        d_Y = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
        GGML_ASSERT(d_Y->size >= y_sz);
    } else {
        d_Y = d_Qy;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline* to_fp16_vk_0 = ggml_vk_get_to_fp16(src0->type);
    vk_pipeline* to_fp16_vk_1 = ggml_vk_get_to_fp16(src1->type);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    // Allocate descriptor sets
    ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne02 * ne03);
    if (qx_needs_dequant) {
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_0, ne02 * ne03);
    }
    if (qy_needs_dequant) {
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_1, ne02 * ne03);
    }
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, ne02 * ne03);
    }

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint32_t it_idx = (i03 * ne02 + i02);
            const uint32_t qx_offset = load_x ? qx_sz * it_idx : 0;
            const uint32_t qy_offset = load_y ? qy_sz * it_idx : 0;
            const uint32_t x_offset = x_sz * it_idx;
            const uint32_t y_offset = y_sz * it_idx;
            const uint32_t d_offset = d_sz * it_idx;

            vk::Semaphore s_x;
            vk::Semaphore s_y;
            vk::Semaphore s_q;

            const vk::Semaphore s_mm = ggml_vk_create_semaphore(compq);

            std::vector<vk::Semaphore> q_semaphores;
            std::vector<vk::Semaphore> mm_semaphores;

            if (load_x) {
                s_x = ggml_vk_create_semaphore(tr0q);
                if (qx_needs_dequant) {
                    q_semaphores.push_back(s_x);
                } else {
                    mm_semaphores.push_back(s_x);
                }
                extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Qx, qx_offset, src0, i03, i02, tr0q, {}, { s_x }, nullptr, &extra->memcpys));
            }
            if (load_y) {
                s_y = ggml_vk_create_semaphore(tr1q);
                if (qy_needs_dequant) {
                    q_semaphores.push_back(s_y);
                } else {
                    mm_semaphores.push_back(s_y);
                }
                extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Qy, qy_offset, src1, i03, i02, tr1q, {}, { s_y }, nullptr, &extra->memcpys));
            }

            if (dq) {
                s_q = ggml_vk_create_semaphore(tr0q);
                vk_submission s = ggml_vk_begin_submission(compq);
                if (qx_needs_dequant) {
                    const std::vector<int> pc = { (int)ne01, (int)ne10, (int)ne10, (int)ne10 };
                    ggml_vk_sync_buffers(s.buffer, { { *d_Qx, qx_offset, qx_sz } }, compq, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                    ggml_vk_sync_buffers(s.buffer, { { *d_X, x_offset, x_sz } }, compq, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite, false);
                    ggml_vk_dispatch_pipeline(s, *to_fp16_vk_0, { { *d_Qx, qx_offset, qx_sz }, { *d_X, x_offset, x_sz } }, pc.size() * sizeof(int), pc.data(), { (uint32_t)x_ne, 1, 1});
                }

                if (qy_needs_dequant) {
                    const std::vector<int> pc = { (int)ne11, (int)ne10, (int)ne10, (int)ne10 };
                    ggml_vk_sync_buffers(s.buffer, { { *d_Qy, qy_offset, qy_sz } }, compq, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                    ggml_vk_sync_buffers(s.buffer, { { *d_Y, y_offset, y_sz } }, compq, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eShaderWrite, false);
                    ggml_vk_dispatch_pipeline(s, *to_fp16_vk_1, { { *d_Qy, qy_offset, qy_sz }, { *d_Y, y_offset, y_sz } }, pc.size() * sizeof(int), pc.data(), { (uint32_t)y_ne, 1, 1});
                }
                ggml_vk_end_submission(s, std::move(q_semaphores), { s_q });
                extra->comp_seqs.push_back({ s });

                mm_semaphores.push_back(s_q);
            }

            // compute
            extra->comp_seqs.push_back(ggml_vk_matmul(*pipeline, { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz }, ne01, ne11, ne10, ne10, ne10, ne01, split_k, compq, std::move(mm_semaphores), { s_mm }));

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, tr1q, { s_mm }, {}));
        }
    }
}

static void ggml_vk_mul_mat_vec_q_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_mul_mat_vec_q_f16((type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3];
    std::cerr << "), (type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3];
    std::cerr << "), (type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << "),)" << std::endl;
#endif
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    GGML_ASSERT(ne11 == 1);

    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    vk_queue& compq = vk_device.compute_queue;
    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qy_needs_dequant = src1->type != GGML_TYPE_F16 && !f16_f32_kernel;

    const bool load_x = src0->backend != GGML_BACKEND_GPU;
    const bool load_y = src1->backend != GGML_BACKEND_GPU;

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const uint32_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t qy_sz = ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    uint32_t buffer_idx = 0;

    vk_buffer* d_D = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    vk_buffer* d_Qx;
    vk_buffer* d_Qy;
    vk_buffer* d_Y;
    if (load_x) {
        d_Qx = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    } else {
        d_Qx = (vk_buffer *) src0->data;
    }
    if (load_y) {
        d_Qy = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    } else {
        d_Qy = (vk_buffer *) src1->data;
    }
    if (qy_needs_dequant) {
        d_Y = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
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
        ggml_vk_pipeline_allocate_descriptor_sets(*to_fp16_vk_1, ne02 * ne03);
    }
    ggml_vk_pipeline_allocate_descriptor_sets(*dmmv, ne02 * ne03);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint32_t it_idx = (i03 * ne02 + i02);
            const uint32_t qx_offset = load_x ? qx_sz * it_idx : 0;
            const uint32_t qy_offset = load_y ? qy_sz * it_idx : 0;
            const uint32_t y_offset = y_sz * it_idx;
            const uint32_t d_offset = d_sz * it_idx;

            vk_submission s = ggml_vk_begin_submission(compq);

            if (load_x) {
                ggml_vk_h2d_tensor_2d(d_Qx, qx_offset, src0, i03, i02, compq, {}, {}, &s, &extra->memcpys);
            }
            if (load_y) {
                ggml_vk_h2d_tensor_2d(d_Qy, qy_offset, src1, i03, i02, compq, {}, {}, &s, &extra->memcpys);
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

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            ggml_vk_sync_buffers(s.buffer, { { *d_D, d_offset, d_sz } }, compq, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead, true);
            ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * d_ne, compq, {}, {}, &s);

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

static vk_pipeline* ggml_vk_op_get_pipeline(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_ADD:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_add_f32;
        } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return &vk_pipeline_add_f16_f32_f16;
        }
        return nullptr;
    case GGML_OP_MUL:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_mul_f32;
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return &vk_pipeline_scale_f32;
        }
        return nullptr;
    default:
        return nullptr;
    }
}

static void ggml_vk_op_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_op op, float scale=1.0f) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_op_f32((type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3];
    std::cerr << "), (type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3];
    std::cerr << "), (type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << "), " << ggml_op_name(op) << ")" << std::endl;
#endif
    GGML_ASSERT(src0->data != nullptr && src1->data != nullptr && dst->data != nullptr);  // NOLINT
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(src1->type));  // NOLINT
    GGML_ASSERT(dst->extra != nullptr);
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne0 = ne00 * ne01 * ne02 * ne03;
    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const int64_t ne1 = ne10 * ne11 * ne12 * ne13;
    const int64_t nb10 = use_src1 ? src1->nb[0] : 0;
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    GGML_ASSERT(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3] == ne0);
    GGML_ASSERT(nb10 == sizeof(float));

    vk_pipeline* pipeline = ggml_vk_op_get_pipeline(src0, src1, dst, op);

    GGML_ASSERT(pipeline != nullptr);

    const bool transfer_src0 = src0->backend != GGML_BACKEND_GPU;
    const bool transfer_src1 = use_src1 && src1->backend != GGML_BACKEND_GPU;

    const uint32_t x_sz = ggml_vk_align_size(ggml_type_size(src0->type) * ne0, vk_device.properties.limits.minStorageBufferOffsetAlignment);
    const uint32_t y_sz = use_src1 ? ggml_vk_align_size(ggml_type_size(src1->type) * ne1, vk_device.properties.limits.minStorageBufferOffsetAlignment) : 0;
    const uint32_t d_sz = ggml_vk_align_size(ggml_type_size(dst->type) * ne0, vk_device.properties.limits.minStorageBufferOffsetAlignment);

    ggml_vk_tensor_extra_gpu * extra = (ggml_vk_tensor_extra_gpu *) dst->extra;

    GGML_ASSERT(extra->comp_seqs.empty());

    uint32_t buffer_idx = 0;

    vk_buffer* d_D = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    vk_buffer* d_X;
    vk_buffer* d_Y;
    if (transfer_src0) {
        d_X = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx++]];
    } else {
        d_X = (vk_buffer *) src0->data;
    }
    if (transfer_src1) {
        d_Y = &vk_preallocated_buffers[extra->buffer_idx[buffer_idx]];
    } else if (use_src1) {
        d_Y = (vk_buffer *) src1->data;
    }

    // Allocate descriptor sets
    ggml_vk_pipeline_allocate_descriptor_sets(*pipeline, ne02 * ne03);

    vk_op_push_constants pc = { (int)ne00, (int)ne01, (int)ne00, (int)ne00, (int)ne00, 0, 0, 0, scale };

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            const uint32_t it_idx = (i03 * ne02 + i02);

            const uint32_t x_offset = transfer_src0 ? x_sz * it_idx : 0;
            const uint32_t y_offset = transfer_src1 ? y_sz * it_idx : 0;
            const uint32_t d_offset = d_sz * it_idx;

            vk::Semaphore s_x;
            vk::Semaphore s_y;
            vk::Semaphore s_mm = ggml_vk_create_semaphore(vk_device.compute_queue);
            std::vector<vk::Semaphore> transfer_semaphores;
            // copy src0 to device
            if (transfer_src0) {
                s_x = ggml_vk_create_semaphore(vk_device.transfer_queues[0]);
                extra->in0_seqs.push_back(ggml_vk_h2d_tensor_2d(d_X, x_offset, src0, i03, i02, vk_device.transfer_queues[0], {}, { s_x }, nullptr, &extra->memcpys));
                transfer_semaphores.push_back(s_x);
            }
            if (transfer_src1) {
                s_y = ggml_vk_create_semaphore(vk_device.transfer_queues[1]);
                extra->in1_seqs.push_back(ggml_vk_h2d_tensor_2d(d_Y, y_offset, src1, i03, i02, vk_device.transfer_queues[1], {}, { s_y }, nullptr, &extra->memcpys));
                transfer_semaphores.push_back(s_y);
            }

            const int64_t i13 = i03%ne13;
            const int64_t i12 = i02%ne12;
            pc.y_offset = (i13*ne12*ne11 + i12*ne11) * ne10;

            vk_submission s = ggml_vk_begin_submission(vk_device.compute_queue);
            ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_D) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, false);
            if (use_src1) {
                ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_X), ggml_vk_subbuffer(*d_Y) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                ggml_vk_dispatch_pipeline(s, *pipeline, { { *d_X, x_offset, x_sz }, { *d_Y, y_offset, y_sz }, { *d_D, d_offset, d_sz } }, sizeof(vk_op_push_constants), &pc, { (uint32_t)ne00, (uint32_t)ne01, 1});
            } else {
                ggml_vk_sync_buffers(s.buffer, { ggml_vk_subbuffer(*d_X) }, vk_device.compute_queue, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, false);
                ggml_vk_dispatch_pipeline(s, *pipeline, { { *d_X, x_offset, x_sz }, { *d_D, d_offset, d_sz } }, sizeof(vk_op_push_constants), &pc, { (uint32_t)ne00, (uint32_t)ne01, 1});
            }
            ggml_vk_end_submission(s, { s_x }, { s_mm });
            extra->comp_seqs.push_back({ s });

            // copy dst to host
            float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);
            extra->out_seqs.push_back(ggml_vk_buffer_read_async(d_D, d_offset, d, sizeof(float) * ne00 * ne01, vk_device.transfer_queues[1], { s_mm }, {}));
        }
    }
}

static void ggml_vk_add(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_ADD);
}

static void ggml_vk_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_MUL);
}

static void ggml_vk_scale(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    ggml_vk_op_f32(src0, src1, dst, GGML_OP_SCALE, ((float *)src1->data)[0]);
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
    seqs.push_back(ggml_vk_h2d_tensor_2d(&dst, 0, tensor, 0, 0, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seqs, VK_NULL_HANDLE);
    vk_device.transfer_queues[0].queue.waitIdle();

    tensor->data = malloc(sizeof(vk_buffer));
    *(vk_buffer*) tensor->data = dst;
    GGML_ASSERT(tensor->backend == GGML_BACKEND_GPU);
}

void ggml_vk_preallocate_buffers_graph(ggml_tensor * node){
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_preallocate_buffers_graph(" << node << ")" << std::endl;
#endif
    node->extra = nullptr;

    const bool any_on_device = node->backend == GGML_BACKEND_GPU
        || (node->src[0] != nullptr && (node->src[0]->backend == GGML_BACKEND_GPU || node->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (node->src[1] != nullptr && node->src[1]->backend == GGML_BACKEND_GPU);

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    const bool use_src0 = src0 != nullptr;
    const int64_t ne00 = use_src0 ? src0->ne[0] : 0;
    const int64_t ne01 = use_src0 ? src0->ne[1] : 0;
    const int64_t ne02 = use_src0 ? src0->ne[2] : 0;
    const int64_t ne03 = use_src0 ? src0->ne[3] : 0;
    const int64_t ne0 = ne00 * ne01 * ne02 * ne03;
    const bool use_src1 = src1 != nullptr;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const int64_t ne1 = ne10 * ne11 * ne12 * ne13;
    const int64_t ne20 = node->ne[0];
    const int64_t ne21 = node->ne[1];
    const int64_t ne22 = node->ne[2];
    const int64_t ne23 = node->ne[3];
    const int64_t ne2 = ne20 * ne21 * ne22 * ne23;

    const bool transfer_src0 = use_src0 && src0->backend != GGML_BACKEND_GPU;
    const bool transfer_src1 = use_src1 && src1->backend != GGML_BACKEND_GPU;

    const bool qvec_kernel = use_src0 && use_src1 && src1->ne[1] == 1 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));
    const bool qx_needs_dequant = use_src0 && !qvec_kernel && src0->type != GGML_TYPE_F16;
    const bool f16_f32_kernel = use_src1 && src1->type == GGML_TYPE_F32;
    const bool qy_needs_dequant = use_src1 && src1->type != GGML_TYPE_F16 && !f16_f32_kernel;
    const bool dq = qx_needs_dequant || qy_needs_dequant;

    const int split_k = node->op == GGML_OP_MUL_MAT ? ggml_vk_guess_split_k(ne01, ne11, ne10) : 1;
    const uint32_t x_ne = ne00 * ne01;
    const uint32_t y_ne = ne10 * ne11;
    const uint32_t d_ne = ne20 * ne21;

    const uint32_t qx_sz = use_src0 ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne02 * ne03 : 0;
    const uint32_t qy_sz = use_src1 ? ggml_vk_align_size(ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type), vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne12 * ne13 : 0;
    const uint32_t x_sz = use_src0 ? ggml_vk_align_size(sizeof(ggml_fp16_t) * x_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne02 * ne03 : 0;
    const uint32_t y_sz = use_src1 ? ggml_vk_align_size(f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne12 * ne13 : 0;
    const uint32_t d_sz = ggml_vk_align_size(sizeof(float) * d_ne * split_k, vk_device.properties.limits.minStorageBufferOffsetAlignment) * ne22 * ne23;

    ggml_vk_tensor_extra_gpu * extra;

    uint32_t idx = 0;

    switch (node->op) {
    case GGML_OP_MUL:
        if (!any_on_device) {
            return;
        }

        extra = new ggml_vk_tensor_extra_gpu;
        extra->buffer_idx.push_back(idx);
        // Check if buffer already exists, increase size if required
        if (idx >= vk_preallocated_buffer_sizes.size()) {
            vk_preallocated_buffer_sizes.push_back(d_sz);
        } else if (vk_preallocated_buffer_sizes[idx] < d_sz) {
            vk_preallocated_buffer_sizes[idx] = d_sz;
        }
        idx++;
        if (transfer_src0) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(qx_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < qx_sz) {
                vk_preallocated_buffer_sizes[idx] = qx_sz;
            }
            idx++;
        }
        if (transfer_src1) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(qy_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < qy_sz) {
                vk_preallocated_buffer_sizes[idx] = qy_sz;
            }
        }
        node->extra = extra;
        vk_gc.extras.push_back(extra);
        break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(node->src[0], node->src[1], node)) {
            return;
        }

        extra = new ggml_vk_tensor_extra_gpu;
        extra->buffer_idx.push_back(idx);
        if (idx >= vk_preallocated_buffer_sizes.size()) {
            vk_preallocated_buffer_sizes.push_back(d_sz);
        } else if (vk_preallocated_buffer_sizes[idx] < d_sz) {
            vk_preallocated_buffer_sizes[idx] = d_sz;
        }
        idx++;
        if (transfer_src0) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(qx_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < qx_sz) {
                vk_preallocated_buffer_sizes[idx] = qx_sz;
            }
            idx++;
        }
        if (transfer_src1) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(qy_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < qy_sz) {
                vk_preallocated_buffer_sizes[idx] = qy_sz;
            }
            idx++;
        }
        if (qx_needs_dequant) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(x_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < x_sz) {
                vk_preallocated_buffer_sizes[idx] = x_sz;
            }
            idx++;
        }
        if (qy_needs_dequant) {
            extra->buffer_idx.push_back(idx);
            if (idx >= vk_preallocated_buffer_sizes.size()) {
                vk_preallocated_buffer_sizes.push_back(y_sz);
            } else if (vk_preallocated_buffer_sizes[idx] < y_sz) {
                vk_preallocated_buffer_sizes[idx] = y_sz;
            }
            idx++;
        }
        node->extra = extra;
        vk_gc.extras.push_back(extra);

        break;
    default:
        break;
    }
}

void ggml_vk_preallocate_buffers() {
    for (size_t i = 0; i < vk_preallocated_buffer_sizes.size(); i++) {
        if (i >= vk_preallocated_buffers.size()) {
            vk_preallocated_buffers.push_back(ggml_vk_create_buffer(vk_preallocated_buffer_sizes[i], vk::MemoryPropertyFlagBits::eDeviceLocal));
        } else if (vk_preallocated_buffers[i].size < vk_preallocated_buffer_sizes[i]) {
            // Resize buffer
            ggml_vk_destroy_buffer(vk_preallocated_buffers[i]);
            vk_preallocated_buffers[i] = ggml_vk_create_buffer(vk_preallocated_buffer_sizes[i], vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    }
}

void ggml_vk_build_graph(ggml_tensor * node){
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_build_graph(" << node << ")" << std::endl;
#endif
    ggml_vk_func_t func;
    const bool any_on_device = node->backend == GGML_BACKEND_GPU
        || (node->src[0] != nullptr && (node->src[0]->backend == GGML_BACKEND_GPU || node->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (node->src[1] != nullptr && node->src[1]->backend == GGML_BACKEND_GPU);

    switch (node->op) {
    // case GGML_OP_ADD:
    //     if (!any_on_device) {
    //         return false;
    //     }

    //     func = ggml_vk_add;

    //     break;
    case GGML_OP_MUL:
        if (!any_on_device) {
            return;
        }

        ggml_vk_mul(node->src[0], node->src[1], node);

        break;
    // case GGML_OP_SCALE:
    //     if (!any_on_device) {
    //         return false;
    //     }

    //     func = ggml_vk_scale;

    //     break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(node->src[0], node->src[1], node)) {
            return;
        }

        ggml_vk_mul_mat(node->src[0], node->src[1], node);

        break;
    default:
        return;
    }
}

bool ggml_vk_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor){
    ggml_vk_func_t func = nullptr;
    const bool any_on_device = tensor->backend == GGML_BACKEND_GPU
        || (tensor->src[0] != nullptr && (tensor->src[0]->backend == GGML_BACKEND_GPU || tensor->src[0]->backend == GGML_BACKEND_GPU_SPLIT))
        || (tensor->src[1] != nullptr && tensor->src[1]->backend == GGML_BACKEND_GPU);

    ggml_vk_tensor_extra_gpu * extra = nullptr;

    switch (tensor->op) {
    // case GGML_OP_ADD:
    //     if (!any_on_device) {
    //         return false;
    //     }

    //     func = ggml_vk_add;

    //     break;
    case GGML_OP_MUL:
        if (!any_on_device) {
            return false;
        }

        extra = (ggml_vk_tensor_extra_gpu *) tensor->extra;

        break;
    // case GGML_OP_SCALE:
    //     if (!any_on_device) {
    //         return false;
    //     }

    //     func = ggml_vk_scale;

    //     break;
    case GGML_OP_MUL_MAT:
        if (!any_on_device && !ggml_vk_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
            return false;
        }

        extra = (ggml_vk_tensor_extra_gpu *) tensor->extra;

        break;
    default:
        return false;
    }

    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }

    GGML_ASSERT(extra);

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

    ggml_vk_queue_cleanup(vk_device.compute_queue);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[0]);
    ggml_vk_queue_cleanup(vk_device.transfer_queues[1]);

    for (auto * extra : vk_gc.extras) {
        delete extra;
    }

    vk_gc.extras.clear();
}

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
        p = &vk_pipeline_matmul_f32_s;
        shname = "F32_S";
    } else if (shader_size == 1) {
        p = &vk_pipeline_matmul_f32_m;
        shname = "F32_M";
    } else if (shader_size == 2) {
        p = &vk_pipeline_matmul_f32_l;
        shname = "F32_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    ggml_vk_pipeline_allocate_descriptor_sets(*p, num_it);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, num_it);
    }

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(float) * kpad * m, &d_X, {});
    ggml_vk_pool_malloc(sizeof(float) * kpad * n, &d_Y, {});
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

    seq.push_back(ggml_vk_buffer_write_2d_async_zeropad(&d_X, 0, x, sizeof(float) * k, sizeof(float) * k, m, sizeof(float) * p->align, vk_device.transfer_queues[0], {}, {}));
    seq.push_back(ggml_vk_buffer_write_2d_async_zeropad(&d_Y, 0, y, sizeof(float) * k, sizeof(float) * k, n, sizeof(float) * p->align, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    // Wait for transfers to finish
    vk_device.transfer_queues[0].queue.waitIdle();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_it; i++) {
        seq.push_back(ggml_vk_matmul(*p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), m, n, k, kpad, kpad, m, split_k, vk_device.compute_queue, {}, {}));
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
        p = &vk_pipeline_matmul_f16_s;
        shname = "F16_S";
    } else if (shader_size == 1) {
        p = &vk_pipeline_matmul_f16_m;
        shname = "F16_M";
    } else if (shader_size == 2) {
        p = &vk_pipeline_matmul_f16_l;
        shname = "F16_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    ggml_vk_pipeline_allocate_descriptor_sets(*p, num_it);
    if (split_k > 1) {
        ggml_vk_pipeline_allocate_descriptor_sets(vk_pipeline_matmul_split_k_reduce, num_it);
    }

    vk_buffer d_X;
    vk_buffer d_Y;
    vk_buffer d_D;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * kpad * m, &d_X, {});
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * kpad * n, &d_Y, {});
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

    seq.push_back(ggml_vk_buffer_write_2d_async_zeropad(&d_X, 0, x, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, m, sizeof(ggml_fp16_t) * p->align, vk_device.transfer_queues[0], {}, {}));
    seq.push_back(ggml_vk_buffer_write_2d_async_zeropad(&d_Y, 0, y, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, n, sizeof(ggml_fp16_t) * p->align, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    // Wait for transfers to finish
    vk_device.transfer_queues[0].queue.waitIdle();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_it; i++) {
        seq.push_back(ggml_vk_matmul(*p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), m, n, k, kpad, kpad, m, split_k, vk_device.compute_queue, {}, {}));
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

void ggml_vk_test_buffer_write_zeropad(size_t m, size_t k, size_t align) {
#ifdef VK_DEBUG
    std::cerr << "ggml_vk_test_buffer_write_zeropad(" << m << ", " << k << ", " << align << ")" << std::endl;
#endif
    std::vector<vk_sequence> seq;

    const size_t kpad = ggml_vk_align_size(k, align);

    vk_buffer d_X;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * kpad * m, &d_X, {});
    vk_buffer d_X2;
    ggml_vk_pool_malloc(sizeof(ggml_fp16_t) * k * m, &d_X2, {});

    ggml_fp16_t* x = (ggml_fp16_t *) ggml_vk_host_malloc(sizeof(ggml_fp16_t) * m * k);

    for (size_t i = 0; i < m * k; i++) {
        x[i] = ggml_fp32_to_fp16(rand() / (float)RAND_MAX);
    }

    seq.push_back(ggml_vk_buffer_write_2d_async_zeropad(&d_X, 0, x, sizeof(ggml_fp16_t) * k, sizeof(ggml_fp16_t) * k, m, sizeof(ggml_fp16_t) * align, vk_device.transfer_queues[0], {}, {}));

    ggml_vk_submit(vk_device.transfer_queues[0], seq, VK_NULL_HANDLE);

    ggml_vk_buffer_write(&d_X2, 0, x, sizeof(ggml_fp16_t) * k * m, vk_device.transfer_queues[0]);

    vk_device.transfer_queues[0].queue.waitIdle();

    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * kpad * m);
    ggml_fp16_t * x_chk2 = (ggml_fp16_t *) malloc(sizeof(ggml_fp16_t) * k * m);

    ggml_vk_buffer_read(&d_X, 0, x_chk, sizeof(ggml_fp16_t) * kpad * m, vk_device.transfer_queues[0]);
    ggml_vk_buffer_read(&d_X2, 0, x_chk2, sizeof(ggml_fp16_t) * k * m, vk_device.transfer_queues[0]);

    double avg_err_async = 0.0;
    double avg_err_sync = 0.0;

    for (size_t kidx = 0; kidx < kpad; kidx++) {
        for (size_t midx = 0; midx < m; midx++) {
            if (kidx < k) {
                const float err = std::fabs(ggml_fp16_to_fp32(x[midx * k + kidx]) - ggml_fp16_to_fp32(x_chk[midx * kpad + kidx]));
                const float err2 = std::fabs(ggml_fp16_to_fp32(x[midx * k + kidx]) - ggml_fp16_to_fp32(x_chk2[midx * k + kidx]));
                if (!std::isnan(err)) {
                    avg_err_async += err;
                }
                if (!std::isnan(err2)) {
                    avg_err_sync += err;
                }

                if (err > 0.01f) {
                    std::cerr << "midx=" << midx << " kidx=" << kidx << " x: " << ggml_fp16_to_fp32(x[midx * k + kidx]) << " x_chk: " << ggml_fp16_to_fp32(x_chk[midx * kpad + kidx]) << " x_chk2: " << ggml_fp16_to_fp32(x_chk2[midx * k + kidx]) << std::endl;
                }
            } else {
                const float val = std::fabs(ggml_fp16_to_fp32(x_chk[midx * kpad + kidx]));
                if (val > 0.01f) {
                    std::cerr << "ZEROPAD ERROR midx=" << midx << " kidx=" << kidx << " src0: 0.0 x_chkidx: " << val << std::endl;
                    GGML_ASSERT(false);
                }
                avg_err_async += val;
            }
        }
    }

    std::cerr << "TEST BUFFER WRITE ZEROPAD m=" << m << " k=" << k << " align=" << align << " avg_err_async=" << avg_err_async / (kpad * m) << " avg_err_sync=" << avg_err_sync / (k * m) << std::endl;

    free(x_chk);
    ggml_vk_host_free(x);
    ggml_vk_pool_free(d_X);
}
#endif
