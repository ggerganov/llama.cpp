#include "ggml-vulkan.h"

#include <vulkan/vulkan.hpp>
#include "external/vk_mem_alloc.h"

#include <iostream>
#include <fstream>

#include "ggml.h"

// static cl_platform_id platform;
// static cl_device_id device;
// static cl_context context;
// static cl_command_queue queue;
// static cl_program program;
// static cl_kernel kernel_q4_0, kernel_q4_1, kernel_q4_2, kernel_q5_0, kernel_q5_1, kernel_q8_0;
// static cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;
// static size_t cl_size_a = 0, cl_size_qb = 0, cl_size_b = 0, cl_size_c = 0;

vk::Instance instance;
vk::PhysicalDevice physical_device;
vk::Device vk_device;
vk::Pipeline vk_pipeline_matmul;
VmaAllocation vk_buffer_qa_alloc, vk_buffer_a_alloc, vk_buffer_b_alloc, vk_buffer_c_alloc;
vk::Buffer vk_buffer_qa, vk_buffer_a, vk_buffer_b, vk_buffer_c;

void ggml_vk_init(void) {
    char* GGML_VULKAN_DEVICE = getenv("GGML_VULKAN_DEVICE");
    int dev_num = (GGML_VULKAN_DEVICE == NULL ? 0 : atoi(GGML_VULKAN_DEVICE));
    printf("\nInitializing Vulkan...");
    printf("\nAttempting to use: Device=%d\n", dev_num);

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION_1_2 };
    const std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags(), &app_info, layers.size(), layers.data());
    instance = vk::createInstance(instance_create_info);

    physical_device = instance.enumeratePhysicalDevices()[dev_num];
    vk::PhysicalDeviceProperties device_props = physical_device.getProperties();
    std::cout << "Picked: " << device_props.deviceName << std::endl;

    std::vector<vk::QueueFamilyProperties> queue_family_props = physical_device.getQueueFamilyProperties();
    auto prop_it = std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const vk::QueueFamilyProperties& prop)
    {
        return prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    const uint32_t compute_queue_family_index = std::distance(queue_family_props.begin(), prop_it);

    const float queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo device_queue_create_info(vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, &queue_priority);
    vk::DeviceCreateInfo device_create_info(vk::DeviceCreateFlags(), device_queue_create_info);
    vk_device = physical_device.createDevice(device_create_info);

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
    vk::DescriptorSetLayout descriptor_set_layout = vk_device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), descriptor_set_layout);
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

// static void ggml_cl_malloc(size_t req_size, size_t* cur_size, cl_mem_flags flags, cl_mem* buf) {
//     if (req_size <= *cur_size) {
//         return;
//     }
//
//     // Reallocate buffer with enough space
//     if (*cur_size > 0) {
//         clReleaseMemObject(*buf);
//     }
//     cl_int err;
//     *buf = clCreateBuffer(context, flags, req_size, NULL, &err);
//     *cur_size = req_size;
//     CL_CHECK(err, "clCreateBuffer");
// }
//
// void ggml_cl_sgemm_wrapper(
//         const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
//         const int m, const int n, const int k,
//         const float alpha, const void *host_a, const int lda,
//         const float *host_b, const int ldb, const float beta,
//         float *host_c, const int ldc, const int btype) {
//     cl_int err = 0;
//
//     cl_kernel kernel;
//     size_t global = n * k, local, size_qb;
//     bool dequant;
//     cl_block_q5_0* cl_host_b;
//
//     switch (btype) {
//     case GGML_TYPE_F32:
//         dequant = false;
//         break;
//     case GGML_TYPE_Q4_0:
//         dequant = true;
//         kernel = kernel_q4_0;
//         local = 16;
//         size_qb = global * (sizeof(float) + local) / 32;
//         break;
//     case GGML_TYPE_Q4_1:
//         dequant = true;
//         kernel = kernel_q4_1;
//         local = 16;
//         size_qb = global * (sizeof(float) * 2 + local) / 32;
//         break;
//     case GGML_TYPE_Q4_2:
//         dequant = true;
//         kernel = kernel_q4_2;
//         local = 8;
//         size_qb = global * (sizeof(ggml_fp16_t) + local) / 16;
//         break;
//     case GGML_TYPE_Q5_0:
//         dequant = true;
//         kernel = kernel_q5_0;
//         local = 16;
//         // For some reason OpenCL seems to be incapable of working with structs of size 22.
//         // 20 and 24 bytes are fine. Workaround to do the fp16 to fp32 step on CPU...
//         // TODO Find the reason, fix and remove workaround.
//         const block_q5_0* b = (const block_q5_0*) host_b;
//         cl_host_b = (cl_block_q5_0*) malloc(sizeof(cl_block_q5_0) * global / 32);
//         for (size_t i = 0; i < global / 32; i++) {
//             cl_host_b[i].d = ggml_fp16_to_fp32(b[i].d);
//             memcpy(&cl_host_b[i].qh, b[i].qh, sizeof(uint32_t));
//             memcpy(&cl_host_b[i].qs, b[i].qs, QK5_0 / 2);
//         }
//         host_b = (const float*) cl_host_b;
//         size_qb = global * (sizeof(float) + sizeof(uint32_t) + local) / 32;
//         break;
//     case GGML_TYPE_Q5_1:
//         dequant = true;
//         kernel = kernel_q5_1;
//         local = 16;
//         size_qb = global * (sizeof(ggml_fp16_t) * 2 + sizeof(uint32_t) + local) / 32;
//         break;
//     case GGML_TYPE_Q8_0:
//         dequant = true;
//         kernel = kernel_q8_0;
//         local = 32;
//         size_qb = global * (sizeof(float) + local) / 32;
//         break;
//     default:
//         fprintf(stderr, "Error: Unsupported OpenCL btype %d\n", btype);
//         abort();
//     }
//
//     const size_t size_a =  m * k * sizeof(float);
//     const size_t size_b =  n * k * sizeof(float);
//     const size_t size_c =  m * n * sizeof(float);
//
//     // Prepare buffers
//     ggml_cl_malloc(size_a, &cl_size_a, CL_MEM_READ_ONLY, &cl_buffer_a);
//     if (dequant) {
//         ggml_cl_malloc(size_qb, &cl_size_qb, CL_MEM_READ_ONLY, &cl_buffer_qb);
//     }
//     ggml_cl_malloc(size_b, &cl_size_b, CL_MEM_READ_WRITE, &cl_buffer_b);
//     ggml_cl_malloc(size_c, &cl_size_c, CL_MEM_WRITE_ONLY, &cl_buffer_c);
//
//     cl_event ev_a, ev_qb, ev_b;
//
//     if (dequant) {
//         err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
//         err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
//         CL_CHECK(err, "clSetKernelArg");
//         err = clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_FALSE, 0, size_qb, host_b, 0, NULL, &ev_qb);
//         CL_CHECK(err, "clEnqueueWriteBuffer qb");
//     } else {
//         err = clEnqueueWriteBuffer(queue, cl_buffer_b, CL_FALSE, 0, size_b, host_b, 0, NULL, &ev_b);
//         CL_CHECK(err, "clEnqueueWriteBuffer b");
//     }
//
//     err = clEnqueueWriteBuffer(queue, cl_buffer_a, CL_FALSE, 0, size_a, host_a, 0, NULL, &ev_a);
//     CL_CHECK(err, "clEnqueueWriteBuffer a");
//     if (dequant) {
//         err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, &ev_qb, &ev_b);
//         CL_CHECK(err, "clEnqueueNDRangeKernel");
//         clReleaseEvent(ev_qb);
//     }
//     clWaitForEvents(1, &ev_a);
//     clWaitForEvents(1, &ev_b);
//     clReleaseEvent(ev_a);
//     clReleaseEvent(ev_b);
//
//     cl_event ev_sgemm;
//     CLBlastStatusCode status = CLBlastSgemm((CLBlastLayout)order,
//                                             (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
//                                             m, n, k,
//                                             alpha,
//                                             cl_buffer_a, 0, lda,
//                                             cl_buffer_b, 0, ldb,
//                                             beta,
//                                             cl_buffer_c, 0, ldc,
//                                             &queue, &ev_sgemm);
//
//     if (status != CLBlastSuccess) {
//         fprintf(stderr, "Error: CLBlast SGEMM %d\n", status);
//         abort();
//     }
//
//     cl_event ev_c;
//     clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c, 1, &ev_sgemm, &ev_c);
//
//     // Wait for completion
//     clWaitForEvents(1, &ev_c);
//     clReleaseEvent(ev_sgemm);
//     clReleaseEvent(ev_c);
//     if (btype == GGML_TYPE_Q5_0) {
//         free((void*) cl_host_b);
//     }
// }
