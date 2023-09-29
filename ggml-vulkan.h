/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <string>

struct ggml_kompute_context;

namespace vk {
    class DeviceMemory;
    class Buffer;
};

struct ggml_vk_memory {
    void *data = nullptr;
    size_t size = 0;
    vk::DeviceMemory *primaryMemory = nullptr;
    vk::Buffer *primaryBuffer = nullptr;
    vk::DeviceMemory *stagingMemory = nullptr;
    vk::Buffer *stagingBuffer = nullptr;
};

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
struct ggml_kompute_context * ggml_vk_init(void);
bool ggml_vk_has_h2d_all(struct ggml_kompute_context * ctx);
void ggml_vk_free(struct ggml_kompute_context * ctx);
size_t ggml_vk_aligned_offset(size_t offset);
ggml_vk_memory ggml_vk_allocate(size_t size);
void ggml_vk_free_memory(ggml_vk_memory &memory);

void ggml_vk_add_buffer(
    struct ggml_kompute_context * ctx,
    const char * name,
    const ggml_vk_memory &memory);

void ggml_vk_h2d_all(struct ggml_kompute_context * ctx);
void ggml_vk_d2h_all(struct ggml_kompute_context * ctx);
void ggml_vk_h2d_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t);
void ggml_vk_d2h_tensor(struct ggml_kompute_context * ctx, struct ggml_tensor * t);
void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf);
