// SPDX-License-Identifier: Apache-2.0

/**
 * Copyright (c) 2023 Nomic, Inc. All rights reserved.
 *
 * This software is licensed under the terms of the Software for Open Models License (SOM),
 * version 1.0, as detailed in the LICENSE_SOM.txt file. A copy of this license should accompany
 * this software. Except as expressly granted in the SOM license, all rights are reserved by Nomic, Inc.
 */

#include "kompute/Core.hpp"

#ifndef KOMPUTE_VK_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#define KOMPUTE_VK_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
/**
 * Ensures support for dynamic loading of Vulkan functions on Android.
 * Acts as a default store for loaded functions.
 * More information:
 * https://github.com/KhronosGroup/Vulkan-Hpp#vulkan_hpp_default_dispatcher
 **/
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif // !KOMPUTE_VK_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace kp {
} // namespace kp
