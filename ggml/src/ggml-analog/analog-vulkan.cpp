// Vulkan dispatch of the compute shaders of vulkan/, see analog-vulkan.h

#include "analog-vulkan.h"
#include "analog-shader.h"
#include "ggml-impl.h"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if !defined(GGML_ANALOG_VULKAN)

// the build has no Vulkan headers, everything falls back to the mirror of the shaders
bool ggml_analog_vulkan_available(void)      { return false; }
const char * ggml_analog_vulkan_status(void) { return "the Vulkan dispatch was not built"; }

bool ggml_analog_vulkan_dac(
        const float *, float *, uint32_t, uint32_t, float, float) { return false; }

bool ggml_analog_vulkan_program(
        const float *, float *, uint32_t, uint32_t, const ggml_analog_config &, float *, float *, float, float) { return false; }

bool ggml_analog_vulkan_readout(
        const float *, float *, const float *, const float *, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float) { return false; }

#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// the Vulkan loader is opened at runtime so that a build with the headers does not link against a
// Vulkan library: dlopen on posix, LoadLibrary on windows
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static void * analog_dlopen(const char * name) {
    return (void *) LoadLibraryA(name);
}

static void * analog_dlsym(void * handle, const char * name) {
    return (void *) GetProcAddress((HMODULE) handle, name);
}

static const char * analog_dlerror(void) {
    return "LoadLibrary failed";
}
#else
#include <dlfcn.h>

static void * analog_dlopen(const char * name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

static void * analog_dlsym(void * handle, const char * name) {
    return dlsym(handle, name);
}

static const char * analog_dlerror(void) {
    return dlerror();
}
#endif

// ---------------------------------------------------------------------------
// function pointers, the loader is opened with dlopen so that a build with the headers does not
// link against a Vulkan library
// ---------------------------------------------------------------------------

struct analog_vk_api {
    PFN_vkGetInstanceProcAddr                    vkGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr                      vkGetDeviceProcAddr;
    PFN_vkCreateInstance                         vkCreateInstance;
    PFN_vkEnumeratePhysicalDevices               vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties            vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceFeatures              vkGetPhysicalDeviceFeatures;
    PFN_vkGetPhysicalDeviceMemoryProperties      vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkCreateDevice                           vkCreateDevice;
    PFN_vkGetDeviceQueue                         vkGetDeviceQueue;
    PFN_vkCreateShaderModule                     vkCreateShaderModule;
    PFN_vkDestroyShaderModule                    vkDestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout              vkCreateDescriptorSetLayout;
    PFN_vkCreatePipelineLayout                   vkCreatePipelineLayout;
    PFN_vkCreateComputePipelines                 vkCreateComputePipelines;
    PFN_vkCreateDescriptorPool                   vkCreateDescriptorPool;
    PFN_vkAllocateDescriptorSets                 vkAllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets                   vkUpdateDescriptorSets;
    PFN_vkCreateBuffer                           vkCreateBuffer;
    PFN_vkGetBufferMemoryRequirements            vkGetBufferMemoryRequirements;
    PFN_vkAllocateMemory                         vkAllocateMemory;
    PFN_vkBindBufferMemory                       vkBindBufferMemory;
    PFN_vkMapMemory                              vkMapMemory;
    PFN_vkUnmapMemory                            vkUnmapMemory;
    PFN_vkDestroyBuffer                          vkDestroyBuffer;
    PFN_vkFreeMemory                             vkFreeMemory;
    PFN_vkCreateCommandPool                      vkCreateCommandPool;
    PFN_vkAllocateCommandBuffers                 vkAllocateCommandBuffers;
    PFN_vkResetCommandBuffer                     vkResetCommandBuffer;
    PFN_vkBeginCommandBuffer                     vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer                       vkEndCommandBuffer;
    PFN_vkCmdBindPipeline                        vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets                  vkCmdBindDescriptorSets;
    PFN_vkCmdPushConstants                       vkCmdPushConstants;
    PFN_vkCmdDispatch                            vkCmdDispatch;
    PFN_vkCmdPipelineBarrier                     vkCmdPipelineBarrier;
    PFN_vkCreateFence                            vkCreateFence;
    PFN_vkResetFences                            vkResetFences;
    PFN_vkWaitForFences                          vkWaitForFences;
    PFN_vkQueueSubmit                            vkQueueSubmit;
};

template <typename T>
static void analog_vk_load(T & fn, PFN_vkGetInstanceProcAddr get, VkInstance instance, const char * name) {
    fn = (T) get(instance, name);

    if (fn == nullptr) {
        GGML_LOG_ERROR("analog: vulkan: %s is not available\n", name);
    }
}

// ---------------------------------------------------------------------------
// push constants, the layout has to match the std430 blocks of the shaders
// ---------------------------------------------------------------------------

struct analog_vk_pc_dac {
    uint32_t k;
    uint32_t nb;
    float    step;
    float    inv_step;
};

struct analog_vk_pc_program {
    uint32_t k;
    uint32_t nrows;
    uint32_t seed_lo;
    uint32_t seed_hi;
    float    step;
    float    inv_step;
    float    sigma_prog;
    float    drift;
    float    ir_drop;
    uint32_t tile_cols;
};

struct analog_vk_pc_xrms {
    uint32_t k;
    uint32_t ncols;
};

struct analog_vk_pc_readout {
    uint32_t m;
    uint32_t n1;
    uint32_t n2;
    uint32_t n3;
    uint32_t ne2;
    uint32_t ne3;
    float    adc_sigma;
    float    sigma;
    float    adc_step;
    float    adc_rstep;
};

static_assert(sizeof(analog_vk_pc_dac)     == 16, "the layout has to match analog_dac.comp");
static_assert(sizeof(analog_vk_pc_program) == 40, "the layout has to match analog_program.comp");
static_assert(sizeof(analog_vk_pc_xrms)    ==  8, "the layout has to match analog_xrms.comp");
static_assert(sizeof(analog_vk_pc_readout) == 40, "the layout has to match analog_readout.comp");

// ---------------------------------------------------------------------------
// the initialized device
// ---------------------------------------------------------------------------

struct analog_vk_shader {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout      layout     = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;
    VkDescriptorSet       set        = VK_NULL_HANDLE;
};

struct analog_vk_buffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *         mapped = nullptr;
    VkDeviceSize   size   = 0;
};

// one dispatch of one kernel, the stages of a call run in order
struct analog_vk_stage {
    analog_vk_shader *              shader;
    std::vector<analog_vk_buffer *> buffers;
    const void *                    pc;
    uint32_t                        pc_size;
    uint32_t                        groups_x;
};

struct analog_vk_state {
    std::mutex  mutex;
    bool        init_done = false;
    bool        ok        = false;
    std::string status    = "not initialized";

    void *         loader = nullptr;
    analog_vk_api  api    = {};

    VkInstance       instance     = VK_NULL_HANDLE;
    VkPhysicalDevice phys         = VK_NULL_HANDLE;
    VkDevice         device       = VK_NULL_HANDLE;
    VkQueue          queue        = VK_NULL_HANDLE;
    uint32_t         queue_family = 0;

    VkCommandPool      cmd_pool  = VK_NULL_HANDLE;
    VkCommandBuffer    cmd       = VK_NULL_HANDLE;
    VkFence            fence     = VK_NULL_HANDLE;
    VkDescriptorPool   desc_pool = VK_NULL_HANDLE;

    analog_vk_shader dac;
    analog_vk_shader program;
    analog_vk_shader xrms;
    analog_vk_shader readout;
};

static analog_vk_state g_vk;

// the SPIR-V module of a kernel, from the ones the build embedded
static const uint32_t * analog_vk_module(const char * name, size_t * n_words) {
    for (int i = 0; i < ggml_analog_shader_spirv_count(); ++i) {
        if (strcmp(ggml_analog_shader_spirv_name(i), name) == 0) {
            return ggml_analog_shader_spirv_words(i, n_words);
        }
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// device memory
// ---------------------------------------------------------------------------

static bool analog_vk_buffer_new(analog_vk_buffer & buf, VkDeviceSize size) {
    VkBufferCreateInfo info = {};
    info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size        = size;
    info.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (g_vk.api.vkCreateBuffer(g_vk.device, &info, nullptr, &buf.buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req;
    g_vk.api.vkGetBufferMemoryRequirements(g_vk.device, buf.buffer, &req);

    VkPhysicalDeviceMemoryProperties props;
    g_vk.api.vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &props);

    // the passes already start and end in host memory, so the buffers are mapped directly and no
    // staging copies are needed
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    }

    if (type == UINT32_MAX) {
        return false;
    }

    VkMemoryAllocateInfo alloc = {};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = type;

    if (g_vk.api.vkAllocateMemory(g_vk.device, &alloc, nullptr, &buf.memory) != VK_SUCCESS) {
        return false;
    }

    if (g_vk.api.vkBindBufferMemory(g_vk.device, buf.buffer, buf.memory, 0) != VK_SUCCESS) {
        return false;
    }

    if (g_vk.api.vkMapMemory(g_vk.device, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped) != VK_SUCCESS) {
        return false;
    }

    buf.size = size;

    return true;
}

static void analog_vk_buffer_free(analog_vk_buffer & buf) {
    if (buf.mapped != nullptr) {
        g_vk.api.vkUnmapMemory(g_vk.device, buf.memory);
        buf.mapped = nullptr;
    }

    if (buf.buffer != VK_NULL_HANDLE) {
        g_vk.api.vkDestroyBuffer(g_vk.device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
    }

    if (buf.memory != VK_NULL_HANDLE) {
        g_vk.api.vkFreeMemory(g_vk.device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
}

// ---------------------------------------------------------------------------
// one submission of a sequence of dispatches
// ---------------------------------------------------------------------------

static bool analog_vk_run(const std::vector<analog_vk_stage> & stages) {
    if (g_vk.api.vkResetCommandBuffer(g_vk.cmd, 0) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (g_vk.api.vkBeginCommandBuffer(g_vk.cmd, &begin) != VK_SUCCESS) {
        return false;
    }

    // the host wrote the inputs, the shaders are about to read them
    VkMemoryBarrier to_shader = {};
    to_shader.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    to_shader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    g_vk.api.vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &to_shader, 0, nullptr, 0, nullptr);

    for (size_t s = 0; s < stages.size(); ++s) {
        const analog_vk_stage & stage = stages[s];

        if (stage.groups_x == 0) {
            continue;
        }

        // the descriptors are rewritten on every call, the queue is idle at this point
        std::vector<VkDescriptorBufferInfo> infos(stage.buffers.size());
        std::vector<VkWriteDescriptorSet>   writes(stage.buffers.size());

        for (size_t i = 0; i < stage.buffers.size(); ++i) {
            infos[i].buffer = stage.buffers[i]->buffer;
            infos[i].offset = 0;
            infos[i].range  = VK_WHOLE_SIZE;

            writes[i] = {};
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = stage.shader->set;
            writes[i].dstBinding      = (uint32_t) i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo     = &infos[i];
        }

        g_vk.api.vkUpdateDescriptorSets(g_vk.device, (uint32_t) writes.size(), writes.data(), 0, nullptr);

        g_vk.api.vkCmdBindPipeline(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, stage.shader->pipeline);
        g_vk.api.vkCmdBindDescriptorSets(g_vk.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, stage.shader->layout, 0, 1,
                &stage.shader->set, 0, nullptr);

        if (stage.pc_size > 0) {
            g_vk.api.vkCmdPushConstants(g_vk.cmd, stage.shader->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, stage.pc_size, stage.pc);
        }

        g_vk.api.vkCmdDispatch(g_vk.cmd, stage.groups_x, 1, 1);

        if (s + 1 < stages.size()) {
            // the next kernel reads what this one wrote
            VkMemoryBarrier between = {};
            between.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            between.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            between.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            g_vk.api.vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &between, 0, nullptr, 0, nullptr);
        }
    }

    // the results are read back by the host
    VkMemoryBarrier to_host = {};
    to_host.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    g_vk.api.vkCmdPipelineBarrier(g_vk.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &to_host, 0, nullptr, 0, nullptr);

    if (g_vk.api.vkEndCommandBuffer(g_vk.cmd) != VK_SUCCESS) {
        return false;
    }

    VkSubmitInfo submit = {};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &g_vk.cmd;

    g_vk.api.vkResetFences(g_vk.device, 1, &g_vk.fence);

    if (g_vk.api.vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) {
        return false;
    }

    return g_vk.api.vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

// ---------------------------------------------------------------------------
// initialization
// ---------------------------------------------------------------------------

static bool analog_vk_shader_init(analog_vk_shader & sh, const uint32_t * words, size_t n_words, uint32_t n_bindings, uint32_t pc_size) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(n_bindings);

    for (uint32_t i = 0; i < n_bindings; ++i) {
        bindings[i] = {};
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = n_bindings;
    layout_info.pBindings    = bindings.data();

    if (g_vk.api.vkCreateDescriptorSetLayout(g_vk.device, &layout_info, nullptr, &sh.set_layout) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset     = 0;
    range.size       = pc_size;

    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount         = 1;
    pipeline_layout_info.pSetLayouts            = &sh.set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges    = &range;

    if (g_vk.api.vkCreatePipelineLayout(g_vk.device, &pipeline_layout_info, nullptr, &sh.layout) != VK_SUCCESS) {
        return false;
    }

    // one descriptor set per kernel, rewritten by every call
    VkDescriptorSetAllocateInfo alloc = {};
    alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool     = g_vk.desc_pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts        = &sh.set_layout;

    if (g_vk.api.vkAllocateDescriptorSets(g_vk.device, &alloc, &sh.set) != VK_SUCCESS) {
        return false;
    }

    VkShaderModuleCreateInfo module_info = {};
    module_info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = n_words*4;
    module_info.pCode    = words;

    VkShaderModule mod = VK_NULL_HANDLE;

    if (g_vk.api.vkCreateShaderModule(g_vk.device, &module_info, nullptr, &mod) != VK_SUCCESS) {
        return false;
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = mod;
    pipeline_info.stage.pName  = "main";
    pipeline_info.layout       = sh.layout;

    const VkResult res = g_vk.api.vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &sh.pipeline);

    g_vk.api.vkDestroyShaderModule(g_vk.device, mod, nullptr);

    return res == VK_SUCCESS;
}

static bool analog_vk_init_locked(void) {
#if !defined(GGML_ANALOG_SPIRV)
    g_vk.status = "the shaders were not built (no GLSL compiler at build time)";
    return false;
#else
#if defined(_WIN32)
    g_vk.loader = analog_dlopen("vulkan-1.dll");
#else
    g_vk.loader = analog_dlopen("libvulkan.so.1");

    if (g_vk.loader == nullptr) {
        g_vk.loader = analog_dlopen("libvulkan.so");
    }
#endif

    if (g_vk.loader == nullptr) {
        g_vk.status = "no Vulkan loader (vulkan-1.dll or libvulkan.so.1), the passes run on the CPU mirror";
        return false;
    }

    g_vk.api.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) analog_dlsym(g_vk.loader, "vkGetInstanceProcAddr");

    if (g_vk.api.vkGetInstanceProcAddr == nullptr) {
        g_vk.status = "the Vulkan loader has no vkGetInstanceProcAddr";
        return false;
    }

    VkApplicationInfo app = {};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "ggml analog backend";
    app.applicationVersion = 0;
    app.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;

    g_vk.api.vkCreateInstance = (PFN_vkCreateInstance) g_vk.api.vkGetInstanceProcAddr(nullptr, "vkCreateInstance");

    if (g_vk.api.vkCreateInstance == nullptr ||
        g_vk.api.vkCreateInstance(&instance_info, nullptr, &g_vk.instance) != VK_SUCCESS) {
        g_vk.status = "a Vulkan instance could not be created";
        return false;
    }

    analog_vk_load(g_vk.api.vkEnumeratePhysicalDevices,               g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkEnumeratePhysicalDevices");
    analog_vk_load(g_vk.api.vkGetPhysicalDeviceProperties,            g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetPhysicalDeviceProperties");
    analog_vk_load(g_vk.api.vkGetPhysicalDeviceFeatures,              g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetPhysicalDeviceFeatures");
    analog_vk_load(g_vk.api.vkGetPhysicalDeviceMemoryProperties,      g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetPhysicalDeviceMemoryProperties");
    analog_vk_load(g_vk.api.vkGetPhysicalDeviceQueueFamilyProperties, g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    analog_vk_load(g_vk.api.vkCreateDevice,                           g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateDevice");

    if (g_vk.api.vkEnumeratePhysicalDevices == nullptr || g_vk.api.vkCreateDevice == nullptr) {
        g_vk.status = "the Vulkan instance has no device entry points";
        return false;
    }

    uint32_t n_devices = 0;
    g_vk.api.vkEnumeratePhysicalDevices(g_vk.instance, &n_devices, nullptr);

    std::vector<VkPhysicalDevice> devices(n_devices);
    if (n_devices > 0) {
        g_vk.api.vkEnumeratePhysicalDevices(g_vk.instance, &n_devices, devices.data());
    }

    std::string last_reason = "no Vulkan device";

    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props;
        g_vk.api.vkGetPhysicalDeviceProperties(dev, &props);

        if (props.apiVersion < VK_API_VERSION_1_1) {
            last_reason = std::string(props.deviceName) + ": Vulkan 1.1 is required";
            continue;
        }

        VkPhysicalDeviceFeatures features;
        g_vk.api.vkGetPhysicalDeviceFeatures(dev, &features);

        if (features.shaderInt64 != VK_TRUE) {
            last_reason = std::string(props.deviceName) + ": the noise of the array needs shaderInt64";
            continue;
        }

        uint32_t n_families = 0;
        g_vk.api.vkGetPhysicalDeviceQueueFamilyProperties(dev, &n_families, nullptr);

        std::vector<VkQueueFamilyProperties> families(n_families);
        if (n_families > 0) {
            g_vk.api.vkGetPhysicalDeviceQueueFamilyProperties(dev, &n_families, families.data());
        }

        uint32_t family = UINT32_MAX;
        for (uint32_t i = 0; i < n_families; ++i) {
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                family = i;
                break;
            }
        }

        if (family == UINT32_MAX) {
            last_reason = std::string(props.deviceName) + ": no compute queue family";
            continue;
        }

        const float priority = 1.0f;

        VkDeviceQueueCreateInfo queue_info = {};
        queue_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount       = 1;
        queue_info.pQueuePriorities = &priority;

        VkDeviceCreateInfo device_info = {};
        device_info.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos    = &queue_info;

        if (g_vk.api.vkCreateDevice(dev, &device_info, nullptr, &g_vk.device) != VK_SUCCESS) {
            last_reason = std::string(props.deviceName) + ": the device could not be created";
            continue;
        }

        g_vk.phys         = dev;
        g_vk.queue_family = family;
        g_vk.status       = std::string("vulkan: ") + props.deviceName;

        break;
    }

    if (g_vk.device == VK_NULL_HANDLE) {
        g_vk.status = last_reason;
        return false;
    }

    analog_vk_load(g_vk.api.vkGetDeviceProcAddr,              g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetDeviceProcAddr");
    analog_vk_load(g_vk.api.vkCreateShaderModule,             g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateShaderModule");
    analog_vk_load(g_vk.api.vkDestroyShaderModule,            g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkDestroyShaderModule");
    analog_vk_load(g_vk.api.vkCreateDescriptorSetLayout,      g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateDescriptorSetLayout");
    analog_vk_load(g_vk.api.vkCreatePipelineLayout,           g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreatePipelineLayout");
    analog_vk_load(g_vk.api.vkCreateComputePipelines,         g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateComputePipelines");
    analog_vk_load(g_vk.api.vkCreateDescriptorPool,           g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateDescriptorPool");
    analog_vk_load(g_vk.api.vkAllocateDescriptorSets,         g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkAllocateDescriptorSets");
    analog_vk_load(g_vk.api.vkUpdateDescriptorSets,           g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkUpdateDescriptorSets");
    analog_vk_load(g_vk.api.vkCreateBuffer,                   g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateBuffer");
    analog_vk_load(g_vk.api.vkGetBufferMemoryRequirements,    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetBufferMemoryRequirements");
    analog_vk_load(g_vk.api.vkAllocateMemory,                 g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkAllocateMemory");
    analog_vk_load(g_vk.api.vkBindBufferMemory,               g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkBindBufferMemory");
    analog_vk_load(g_vk.api.vkMapMemory,                      g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkMapMemory");
    analog_vk_load(g_vk.api.vkUnmapMemory,                    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkUnmapMemory");
    analog_vk_load(g_vk.api.vkDestroyBuffer,                  g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkDestroyBuffer");
    analog_vk_load(g_vk.api.vkFreeMemory,                     g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkFreeMemory");
    analog_vk_load(g_vk.api.vkCreateCommandPool,              g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateCommandPool");
    analog_vk_load(g_vk.api.vkAllocateCommandBuffers,         g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkAllocateCommandBuffers");
    analog_vk_load(g_vk.api.vkResetCommandBuffer,             g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkResetCommandBuffer");
    analog_vk_load(g_vk.api.vkBeginCommandBuffer,             g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkBeginCommandBuffer");
    analog_vk_load(g_vk.api.vkEndCommandBuffer,               g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkEndCommandBuffer");
    analog_vk_load(g_vk.api.vkCmdBindPipeline,                g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCmdBindPipeline");
    analog_vk_load(g_vk.api.vkCmdBindDescriptorSets,          g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCmdBindDescriptorSets");
    analog_vk_load(g_vk.api.vkCmdPushConstants,               g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCmdPushConstants");
    analog_vk_load(g_vk.api.vkCmdDispatch,                    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCmdDispatch");
    analog_vk_load(g_vk.api.vkCmdPipelineBarrier,             g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCmdPipelineBarrier");
    analog_vk_load(g_vk.api.vkCreateFence,                    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkCreateFence");
    analog_vk_load(g_vk.api.vkResetFences,                    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkResetFences");
    analog_vk_load(g_vk.api.vkWaitForFences,                  g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkWaitForFences");
    analog_vk_load(g_vk.api.vkQueueSubmit,                    g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkQueueSubmit");
    analog_vk_load(g_vk.api.vkGetDeviceQueue,                 g_vk.api.vkGetInstanceProcAddr, g_vk.instance, "vkGetDeviceQueue");

    if (g_vk.api.vkCreateShaderModule == nullptr || g_vk.api.vkCreateComputePipelines == nullptr ||
        g_vk.api.vkCmdDispatch == nullptr || g_vk.api.vkQueueSubmit == nullptr ||
        g_vk.api.vkGetDeviceQueue == nullptr || g_vk.api.vkUnmapMemory == nullptr) {
        g_vk.status = "the Vulkan device has no compute entry points";
        return false;
    }

    g_vk.api.vkGetDeviceQueue(g_vk.device, g_vk.queue_family, 0, &g_vk.queue);

    VkCommandPoolCreateInfo cmd_info = {};
    cmd_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmd_info.queueFamilyIndex = g_vk.queue_family;

    if (g_vk.api.vkCreateCommandPool(g_vk.device, &cmd_info, nullptr, &g_vk.cmd_pool) != VK_SUCCESS) {
        g_vk.status = "no Vulkan command pool";
        return false;
    }

    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool        = g_vk.cmd_pool;
    cmd_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    if (g_vk.api.vkAllocateCommandBuffers(g_vk.device, &cmd_alloc, &g_vk.cmd) != VK_SUCCESS) {
        g_vk.status = "no Vulkan command buffer";
        return false;
    }

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (g_vk.api.vkCreateFence(g_vk.device, &fence_info, nullptr, &g_vk.fence) != VK_SUCCESS) {
        g_vk.status = "no Vulkan fence";
        return false;
    }

    // four kernels, at most four storage buffers each
    VkDescriptorPoolSize size = {};
    size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = 16;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets       = 4;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &size;

    if (g_vk.api.vkCreateDescriptorPool(g_vk.device, &pool_info, nullptr, &g_vk.desc_pool) != VK_SUCCESS) {
        g_vk.status = "no Vulkan descriptor pool";
        return false;
    }

    struct analog_vk_kernel {
        analog_vk_shader * shader;
        const char *       name;
        uint32_t           n_bindings;
        uint32_t           pc_size;
    };

    const analog_vk_kernel kernels[] = {
        { &g_vk.dac,     "analog_dac",     2, sizeof(analog_vk_pc_dac)     },
        { &g_vk.program, "analog_program", 4, sizeof(analog_vk_pc_program) },
        { &g_vk.xrms,    "analog_xrms",    2, sizeof(analog_vk_pc_xrms)    },
        { &g_vk.readout, "analog_readout", 4, sizeof(analog_vk_pc_readout) },
    };

    bool ok = true;

    for (const analog_vk_kernel & kernel : kernels) {
        size_t n_words = 0;

        const uint32_t * words = analog_vk_module(kernel.name, &n_words);

        if (words == nullptr) {
            GGML_LOG_ERROR("analog: vulkan: the SPIR-V module of %s is missing\n", kernel.name);
            ok = false;
            break;
        }

        ok = analog_vk_shader_init(*kernel.shader, words, n_words, kernel.n_bindings, kernel.pc_size);

        if (!ok) {
            GGML_LOG_ERROR("analog: vulkan: %s could not be initialized\n", kernel.name);
            break;
        }
    }

    if (!ok) {
        GGML_LOG_ERROR("analog: vulkan: the compute pipelines could not be created\n");
        g_vk.status = "the compute pipelines could not be created";
        return false;
    }

    GGML_LOG_INFO("analog: vulkan: %s (queue family %u)\n", g_vk.status.c_str(), g_vk.queue_family);

    return true;
#endif
}

static bool analog_vk_ensure(void) {
    std::lock_guard<std::mutex> lock(g_vk.mutex);

    if (!g_vk.init_done) {
        g_vk.ok        = analog_vk_init_locked();
        g_vk.init_done = true;
    }

    return g_vk.ok;
}

// ---------------------------------------------------------------------------
// the kernels
// ---------------------------------------------------------------------------

bool ggml_analog_vulkan_available(void) {
    return analog_vk_ensure();
}

const char * ggml_analog_vulkan_status(void) {
    analog_vk_ensure();

    return g_vk.status.c_str();
}

bool ggml_analog_vulkan_dac(
        const float * x,
              float * q,
        uint32_t k, uint32_t ncols,
        float step, float inv_step) {
    if (!analog_vk_ensure()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_vk.mutex);

    analog_vk_buffer bx;
    analog_vk_buffer bq;

    if (!analog_vk_buffer_new(bx, (VkDeviceSize) k*ncols*sizeof(float)) ||
        !analog_vk_buffer_new(bq, (VkDeviceSize) k*ncols*sizeof(float))) {
        return false;
    }

    memcpy(bx.mapped, x, (size_t) k*ncols*sizeof(float));

    analog_vk_pc_dac pc;

    pc.k        = k;
    pc.nb       = (k + 31)/32;
    pc.step     = step;
    pc.inv_step = inv_step;

    const uint32_t groups = (pc.nb*ncols + 63)/64;

    const bool ok = analog_vk_run({ { &g_vk.dac, { &bx, &bq }, &pc, sizeof(pc), groups } });

    if (ok) {
        memcpy(q, bq.mapped, (size_t) k*ncols*sizeof(float));
    }

    analog_vk_buffer_free(bx);
    analog_vk_buffer_free(bq);

    return ok;
}

bool ggml_analog_vulkan_program(
        const float * w,
              float * snap,
        uint32_t k, uint32_t nrows,
        const ggml_analog_config & cfg,
              float * row_norm,
              float * row_inv_norm,
        float step, float inv_step) {
    if (!analog_vk_ensure()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_vk.mutex);

    analog_vk_buffer bw;
    analog_vk_buffer bs;
    analog_vk_buffer bn;
    analog_vk_buffer bi;

    const VkDeviceSize cells = (VkDeviceSize) k*nrows;

    if (!analog_vk_buffer_new(bw, cells*sizeof(float)) ||
        !analog_vk_buffer_new(bs, cells*sizeof(float)) ||
        !analog_vk_buffer_new(bn, (VkDeviceSize) nrows*sizeof(float)) ||
        !analog_vk_buffer_new(bi, (VkDeviceSize) nrows*sizeof(float))) {
        return false;
    }

    memcpy(bw.mapped, w, (size_t) cells*sizeof(float));

    analog_vk_pc_program pc;

    pc.k          = k;
    pc.nrows      = nrows;
    pc.seed_lo    = (uint32_t) (cfg.seed & 0xFFFFFFFFu);
    pc.seed_hi    = (uint32_t) (cfg.seed >> 32);
    pc.step       = step;
    pc.inv_step   = inv_step;
    pc.sigma_prog = cfg.sigma_prog;
    pc.drift      = cfg.drift;
    pc.ir_drop    = cfg.ir_drop;
    pc.tile_cols  = (uint32_t) cfg.tile_cols;

    const uint32_t groups = (nrows + 63)/64;

    const bool ok = analog_vk_run({ { &g_vk.program, { &bw, &bs, &bn, &bi }, &pc, sizeof(pc), groups } });

    if (ok) {
        memcpy(snap,         bs.mapped, (size_t) cells*sizeof(float));
        memcpy(row_norm,     bn.mapped, (size_t) nrows*sizeof(float));
        memcpy(row_inv_norm, bi.mapped, (size_t) nrows*sizeof(float));
    }

    analog_vk_buffer_free(bw);
    analog_vk_buffer_free(bs);
    analog_vk_buffer_free(bn);
    analog_vk_buffer_free(bi);

    return ok;
}

bool ggml_analog_vulkan_readout(
        const float * x,
              float * y,
        const float * row_norm,
        const float * row_inv_norm,
        uint32_t k, uint32_t m,
        uint32_t n1, uint32_t n2, uint32_t n3,
        uint32_t ne2, uint32_t ne3,
        float adc_sigma, float sigma, float adc_step, float adc_rstep) {
    if (!analog_vk_ensure()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_vk.mutex);

    const uint32_t ncols = n1*n2*n3;

    analog_vk_buffer bx;
    analog_vk_buffer by;
    analog_vk_buffer br;
    analog_vk_buffer bn;
    analog_vk_buffer bi;

    const VkDeviceSize rows_k = (VkDeviceSize) m*ne2*ne3;

    if (!analog_vk_buffer_new(bx, (VkDeviceSize) k*ncols*sizeof(float)) ||
        !analog_vk_buffer_new(by, (VkDeviceSize) m*ncols*sizeof(float)) ||
        !analog_vk_buffer_new(br, (VkDeviceSize) ncols*sizeof(float)) ||
        !analog_vk_buffer_new(bn, rows_k*sizeof(float)) ||
        !analog_vk_buffer_new(bi, rows_k*sizeof(float))) {
        return false;
    }

    memcpy(bx.mapped, x,            (size_t) k*ncols*sizeof(float));
    memcpy(by.mapped, y,            (size_t) m*ncols*sizeof(float));
    memcpy(bn.mapped, row_norm,     (size_t) rows_k*sizeof(float));
    memcpy(bi.mapped, row_inv_norm, (size_t) rows_k*sizeof(float));

    analog_vk_pc_xrms pc_rms;

    pc_rms.k     = k;
    pc_rms.ncols = ncols;

    analog_vk_pc_readout pc;

    pc.m         = m;
    pc.n1        = n1;
    pc.n2        = n2;
    pc.n3        = n3;
    pc.ne2       = ne2;
    pc.ne3       = ne3;
    pc.adc_sigma = adc_sigma;
    pc.sigma     = sigma;
    pc.adc_step  = adc_step;
    pc.adc_rstep = adc_rstep;

    // the readout reads the RMS of every column, the xrms kernel runs first
    const bool ok = analog_vk_run({
        { &g_vk.xrms,    { &bx, &br },             &pc_rms, sizeof(pc_rms), (ncols + 63)/64 },
        { &g_vk.readout, { &by, &br, &bn, &bi },   &pc,     sizeof(pc),     (ncols*m + 63)/64 },
    });

    if (ok) {
        memcpy(y, by.mapped, (size_t) m*ncols*sizeof(float));
    }

    analog_vk_buffer_free(bx);
    analog_vk_buffer_free(by);
    analog_vk_buffer_free(br);
    analog_vk_buffer_free(bn);
    analog_vk_buffer_free(bi);

    return ok;
}

#endif
