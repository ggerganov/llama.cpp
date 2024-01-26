#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <string>
#include <vector>

struct ggml_vk_device {
    int index = 0;
    int type = 0;           // same as VkPhysicalDeviceType
    size_t heapSize = 0;
    std::string name;
    std::string vendor;
    int subgroupSize = 0;
};

std::vector<ggml_vk_device> ggml_vk_available_devices(size_t memoryRequired);
bool ggml_vk_init_device(size_t memoryRequired, const std::string &device);
bool ggml_vk_init_device(const ggml_vk_device &device);
bool ggml_vk_init_device(int device);
bool ggml_vk_free_device();
bool ggml_vk_has_vulkan();
bool ggml_vk_has_device();
bool ggml_vk_using_vulkan();
ggml_vk_device ggml_vk_current_device();

//
// backend API
// user-code should use only these functions
//

#ifdef __cplusplus
extern "C" {
#endif

// forward declaration
typedef struct ggml_backend * ggml_backend_t;

GGML_API ggml_backend_t ggml_backend_kompute_init(void);

GGML_API bool ggml_backend_is_kompute(ggml_backend_t backend);

GGML_API ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(void);

#ifdef __cplusplus
}
#endif
