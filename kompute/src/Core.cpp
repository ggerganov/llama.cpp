// SPDX-License-Identifier: Apache-2.0

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
