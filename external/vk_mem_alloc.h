//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#ifndef AMD_VULKAN_MEMORY_ALLOCATOR_H
#define AMD_VULKAN_MEMORY_ALLOCATOR_H

/** \mainpage Vulkan Memory Allocator

<b>Version 3.1.0-development</b>

Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved. \n
License: MIT

<b>API documentation divided into groups:</b> [Modules](modules.html)

\section main_table_of_contents Table of contents

- <b>User guide</b>
  - \subpage quick_start
    - [Project setup](@ref quick_start_project_setup)
    - [Initialization](@ref quick_start_initialization)
    - [Resource allocation](@ref quick_start_resource_allocation)
  - \subpage choosing_memory_type
    - [Usage](@ref choosing_memory_type_usage)
    - [Required and preferred flags](@ref choosing_memory_type_required_preferred_flags)
    - [Explicit memory types](@ref choosing_memory_type_explicit_memory_types)
    - [Custom memory pools](@ref choosing_memory_type_custom_memory_pools)
    - [Dedicated allocations](@ref choosing_memory_type_dedicated_allocations)
  - \subpage memory_mapping
    - [Mapping functions](@ref memory_mapping_mapping_functions)
    - [Persistently mapped memory](@ref memory_mapping_persistently_mapped_memory)
    - [Cache flush and invalidate](@ref memory_mapping_cache_control)
  - \subpage staying_within_budget
    - [Querying for budget](@ref staying_within_budget_querying_for_budget)
    - [Controlling memory usage](@ref staying_within_budget_controlling_memory_usage)
  - \subpage resource_aliasing
  - \subpage custom_memory_pools
    - [Choosing memory type index](@ref custom_memory_pools_MemTypeIndex)
    - [Linear allocation algorithm](@ref linear_algorithm)
      - [Free-at-once](@ref linear_algorithm_free_at_once)
      - [Stack](@ref linear_algorithm_stack)
      - [Double stack](@ref linear_algorithm_double_stack)
      - [Ring buffer](@ref linear_algorithm_ring_buffer)
  - \subpage defragmentation
  - \subpage statistics
    - [Numeric statistics](@ref statistics_numeric_statistics)
    - [JSON dump](@ref statistics_json_dump)
  - \subpage allocation_annotation
    - [Allocation user data](@ref allocation_user_data)
    - [Allocation names](@ref allocation_names)
  - \subpage virtual_allocator
  - \subpage debugging_memory_usage
    - [Memory initialization](@ref debugging_memory_usage_initialization)
    - [Margins](@ref debugging_memory_usage_margins)
    - [Corruption detection](@ref debugging_memory_usage_corruption_detection)
  - \subpage opengl_interop
- \subpage usage_patterns
    - [GPU-only resource](@ref usage_patterns_gpu_only)
    - [Staging copy for upload](@ref usage_patterns_staging_copy_upload)
    - [Readback](@ref usage_patterns_readback)
    - [Advanced data uploading](@ref usage_patterns_advanced_data_uploading)
    - [Other use cases](@ref usage_patterns_other_use_cases)
- \subpage configuration
  - [Pointers to Vulkan functions](@ref config_Vulkan_functions)
  - [Custom host memory allocator](@ref custom_memory_allocator)
  - [Device memory allocation callbacks](@ref allocation_callbacks)
  - [Device heap memory limit](@ref heap_memory_limit)
- <b>Extension support</b>
    - \subpage vk_khr_dedicated_allocation
    - \subpage enabling_buffer_device_address
    - \subpage vk_ext_memory_priority
    - \subpage vk_amd_device_coherent_memory
- \subpage general_considerations
  - [Thread safety](@ref general_considerations_thread_safety)
  - [Versioning and compatibility](@ref general_considerations_versioning_and_compatibility)
  - [Validation layer warnings](@ref general_considerations_validation_layer_warnings)
  - [Allocation algorithm](@ref general_considerations_allocation_algorithm)
  - [Features not supported](@ref general_considerations_features_not_supported)

\section main_see_also See also

- [**Product page on GPUOpen**](https://gpuopen.com/gaming-product/vulkan-memory-allocator/)
- [**Source repository on GitHub**](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)

\defgroup group_init Library initialization

\brief API elements related to the initialization and management of the entire library, especially #VmaAllocator object.

\defgroup group_alloc Memory allocation

\brief API elements related to the allocation, deallocation, and management of Vulkan memory, buffers, images.
Most basic ones being: vmaCreateBuffer(), vmaCreateImage().

\defgroup group_virtual Virtual allocator

\brief API elements related to the mechanism of \ref virtual_allocator - using the core allocation algorithm
for user-defined purpose without allocating any real GPU memory.

\defgroup group_stats Statistics

\brief API elements that query current status of the allocator, from memory usage, budget, to full dump of the internal state in JSON format.
See documentation chapter: \ref statistics.
*/


#ifdef __cplusplus
extern "C" {
#endif

#ifndef VULKAN_H_
    #include <vulkan/vulkan.h>
#endif

#if !defined(VMA_VULKAN_VERSION)
    #if defined(VK_VERSION_1_3)
        #define VMA_VULKAN_VERSION 1003000
    #elif defined(VK_VERSION_1_2)
        #define VMA_VULKAN_VERSION 1002000
    #elif defined(VK_VERSION_1_1)
        #define VMA_VULKAN_VERSION 1001000
    #else
        #define VMA_VULKAN_VERSION 1000000
    #endif
#endif

#if defined(__ANDROID__) && defined(VK_NO_PROTOTYPES) && VMA_STATIC_VULKAN_FUNCTIONS
    extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    extern PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
    extern PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    extern PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    extern PFN_vkAllocateMemory vkAllocateMemory;
    extern PFN_vkFreeMemory vkFreeMemory;
    extern PFN_vkMapMemory vkMapMemory;
    extern PFN_vkUnmapMemory vkUnmapMemory;
    extern PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    extern PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    extern PFN_vkBindBufferMemory vkBindBufferMemory;
    extern PFN_vkBindImageMemory vkBindImageMemory;
    extern PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    extern PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
    extern PFN_vkCreateBuffer vkCreateBuffer;
    extern PFN_vkDestroyBuffer vkDestroyBuffer;
    extern PFN_vkCreateImage vkCreateImage;
    extern PFN_vkDestroyImage vkDestroyImage;
    extern PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    #if VMA_VULKAN_VERSION >= 1001000
        extern PFN_vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements2;
        extern PFN_vkGetImageMemoryRequirements2 vkGetImageMemoryRequirements2;
        extern PFN_vkBindBufferMemory2 vkBindBufferMemory2;
        extern PFN_vkBindImageMemory2 vkBindImageMemory2;
        extern PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2;
    #endif // #if VMA_VULKAN_VERSION >= 1001000
#endif // #if defined(__ANDROID__) && VMA_STATIC_VULKAN_FUNCTIONS && VK_NO_PROTOTYPES

#if !defined(VMA_DEDICATED_ALLOCATION)
    #if VK_KHR_get_memory_requirements2 && VK_KHR_dedicated_allocation
        #define VMA_DEDICATED_ALLOCATION 1
    #else
        #define VMA_DEDICATED_ALLOCATION 0
    #endif
#endif

#if !defined(VMA_BIND_MEMORY2)
    #if VK_KHR_bind_memory2
        #define VMA_BIND_MEMORY2 1
    #else
        #define VMA_BIND_MEMORY2 0
    #endif
#endif

#if !defined(VMA_MEMORY_BUDGET)
    #if VK_EXT_memory_budget && (VK_KHR_get_physical_device_properties2 || VMA_VULKAN_VERSION >= 1001000)
        #define VMA_MEMORY_BUDGET 1
    #else
        #define VMA_MEMORY_BUDGET 0
    #endif
#endif

// Defined to 1 when VK_KHR_buffer_device_address device extension or equivalent core Vulkan 1.2 feature is defined in its headers.
#if !defined(VMA_BUFFER_DEVICE_ADDRESS)
    #if VK_KHR_buffer_device_address || VMA_VULKAN_VERSION >= 1002000
        #define VMA_BUFFER_DEVICE_ADDRESS 1
    #else
        #define VMA_BUFFER_DEVICE_ADDRESS 0
    #endif
#endif

// Defined to 1 when VK_EXT_memory_priority device extension is defined in Vulkan headers.
#if !defined(VMA_MEMORY_PRIORITY)
    #if VK_EXT_memory_priority
        #define VMA_MEMORY_PRIORITY 1
    #else
        #define VMA_MEMORY_PRIORITY 0
    #endif
#endif

// Defined to 1 when VK_KHR_external_memory device extension is defined in Vulkan headers.
#if !defined(VMA_EXTERNAL_MEMORY)
    #if VK_KHR_external_memory
        #define VMA_EXTERNAL_MEMORY 1
    #else
        #define VMA_EXTERNAL_MEMORY 0
    #endif
#endif

// Define these macros to decorate all public functions with additional code,
// before and after returned type, appropriately. This may be useful for
// exporting the functions when compiling VMA as a separate library. Example:
// #define VMA_CALL_PRE  __declspec(dllexport)
// #define VMA_CALL_POST __cdecl
#ifndef VMA_CALL_PRE
    #define VMA_CALL_PRE
#endif
#ifndef VMA_CALL_POST
    #define VMA_CALL_POST
#endif

// Define this macro to decorate pointers with an attribute specifying the
// length of the array they point to if they are not null.
//
// The length may be one of
// - The name of another parameter in the argument list where the pointer is declared
// - The name of another member in the struct where the pointer is declared
// - The name of a member of a struct type, meaning the value of that member in
//   the context of the call. For example
//   VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount"),
//   this means the number of memory heaps available in the device associated
//   with the VmaAllocator being dealt with.
#ifndef VMA_LEN_IF_NOT_NULL
    #define VMA_LEN_IF_NOT_NULL(len)
#endif

// The VMA_NULLABLE macro is defined to be _Nullable when compiling with Clang.
// see: https://clang.llvm.org/docs/AttributeReference.html#nullable
#ifndef VMA_NULLABLE
    #ifdef __clang__
        #define VMA_NULLABLE _Nullable
    #else
        #define VMA_NULLABLE
    #endif
#endif

// The VMA_NOT_NULL macro is defined to be _Nonnull when compiling with Clang.
// see: https://clang.llvm.org/docs/AttributeReference.html#nonnull
#ifndef VMA_NOT_NULL
    #ifdef __clang__
        #define VMA_NOT_NULL _Nonnull
    #else
        #define VMA_NOT_NULL
    #endif
#endif

// If non-dispatchable handles are represented as pointers then we can give
// then nullability annotations
#ifndef VMA_NOT_NULL_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define VMA_NOT_NULL_NON_DISPATCHABLE VMA_NOT_NULL
    #else
        #define VMA_NOT_NULL_NON_DISPATCHABLE
    #endif
#endif

#ifndef VMA_NULLABLE_NON_DISPATCHABLE
    #if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__) ) || defined(_M_X64) || defined(__ia64) || defined (_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
        #define VMA_NULLABLE_NON_DISPATCHABLE VMA_NULLABLE
    #else
        #define VMA_NULLABLE_NON_DISPATCHABLE
    #endif
#endif

#ifndef VMA_STATS_STRING_ENABLED
    #define VMA_STATS_STRING_ENABLED 1
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//    INTERFACE
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Sections for managing code placement in file, only for development purposes e.g. for convenient folding inside an IDE.
#ifndef _VMA_ENUM_DECLARATIONS

/**
\addtogroup group_init
@{
*/

/// Flags for created #VmaAllocator.
typedef enum VmaAllocatorCreateFlagBits
{
    /** \brief Allocator and all objects created from it will not be synchronized internally, so you must guarantee they are used from only one thread at a time or synchronized externally by you.

    Using this flag may increase performance because internal mutexes are not used.
    */
    VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT = 0x00000001,
    /** \brief Enables usage of VK_KHR_dedicated_allocation extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    Using this extension will automatically allocate dedicated blocks of memory for
    some buffers and images instead of suballocating place for them out of bigger
    memory blocks (as if you explicitly used #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    flag) when it is recommended by the driver. It may improve performance on some
    GPUs.

    You may set this flag only if you found out that following device extensions are
    supported, you enabled them while creating Vulkan device passed as
    VmaAllocatorCreateInfo::device, and you want them to be used internally by this
    library:

    - VK_KHR_get_memory_requirements2 (device extension)
    - VK_KHR_dedicated_allocation (device extension)

    When this flag is set, you can experience following warnings reported by Vulkan
    validation layer. You can ignore them.

    > vkBindBufferMemory(): Binding memory to buffer 0x2d but vkGetBufferMemoryRequirements() has not been called on that buffer.
    */
    VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT = 0x00000002,
    /**
    Enables usage of VK_KHR_bind_memory2 extension.

    The flag works only if VmaAllocatorCreateInfo::vulkanApiVersion `== VK_API_VERSION_1_0`.
    When it is `VK_API_VERSION_1_1`, the flag is ignored because the extension has been promoted to Vulkan 1.1.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library.

    The extension provides functions `vkBindBufferMemory2KHR` and `vkBindImageMemory2KHR`,
    which allow to pass a chain of `pNext` structures while binding.
    This flag is required if you use `pNext` parameter in vmaBindBufferMemory2() or vmaBindImageMemory2().
    */
    VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT = 0x00000004,
    /**
    Enables usage of VK_EXT_memory_budget extension.

    You may set this flag only if you found out that this device extension is supported,
    you enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    and you want it to be used internally by this library, along with another instance extension
    VK_KHR_get_physical_device_properties2, which is required by it (or Vulkan 1.1, where this extension is promoted).

    The extension provides query for current memory usage and budget, which will probably
    be more accurate than an estimation used by the library otherwise.
    */
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT = 0x00000008,
    /**
    Enables usage of VK_AMD_device_coherent_memory extension.

    You may set this flag only if you:

    - found out that this device extension is supported and enabled it while creating Vulkan device passed as VmaAllocatorCreateInfo::device,
    - checked that `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true and set it while creating the Vulkan device,
    - want it to be used internally by this library.

    The extension and accompanying device feature provide access to memory types with
    `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flags.
    They are useful mostly for writing breadcrumb markers - a common method for debugging GPU crash/hang/TDR.

    When the extension is not enabled, such memory types are still enumerated, but their usage is illegal.
    To protect from this error, if you don't create the allocator with this flag, it will refuse to allocate any memory or create a custom pool in such memory type,
    returning `VK_ERROR_FEATURE_NOT_PRESENT`.
    */
    VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT = 0x00000010,
    /**
    Enables usage of "buffer device address" feature, which allows you to use function
    `vkGetBufferDeviceAddress*` to get raw GPU pointer to a buffer and pass it for usage inside a shader.

    You may set this flag only if you:

    1. (For Vulkan version < 1.2) Found as available and enabled device extension
    VK_KHR_buffer_device_address.
    This extension is promoted to core Vulkan 1.2.
    2. Found as available and enabled device feature `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress`.

    When this flag is set, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` using VMA.
    The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` to
    allocated memory blocks wherever it might be needed.

    For more information, see documentation chapter \ref enabling_buffer_device_address.
    */
    VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT = 0x00000020,
    /**
    Enables usage of VK_EXT_memory_priority extension in the library.

    You may set this flag only if you found available and enabled this device extension,
    along with `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority == VK_TRUE`,
    while creating Vulkan device passed as VmaAllocatorCreateInfo::device.

    When this flag is used, VmaAllocationCreateInfo::priority and VmaPoolCreateInfo::priority
    are used to set priorities of allocated Vulkan memory. Without it, these variables are ignored.

    A priority must be a floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.
    Larger values are higher priority. The granularity of the priorities is implementation-dependent.
    It is automatically passed to every call to `vkAllocateMemory` done by the library using structure `VkMemoryPriorityAllocateInfoEXT`.
    The value to be used for default priority is 0.5.
    For more details, see the documentation of the VK_EXT_memory_priority extension.
    */
    VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT = 0x00000040,

    VMA_ALLOCATOR_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocatorCreateFlagBits;
/// See #VmaAllocatorCreateFlagBits.
typedef VkFlags VmaAllocatorCreateFlags;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/// \brief Intended usage of the allocated memory.
typedef enum VmaMemoryUsage
{
    /** No intended memory usage specified.
    Use other members of VmaAllocationCreateInfo to specify your requirements.
    */
    VMA_MEMORY_USAGE_UNKNOWN = 0,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_GPU_ONLY = 1,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` and `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_ONLY = 2,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_TO_GPU = 3,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Guarantees `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, prefers `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.
    */
    VMA_MEMORY_USAGE_GPU_TO_CPU = 4,
    /**
    \deprecated Obsolete, preserved for backward compatibility.
    Prefers not `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
    */
    VMA_MEMORY_USAGE_CPU_COPY = 5,
    /**
    Lazily allocated GPU memory having `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`.
    Exists mostly on mobile platforms. Using it on desktop PC or other GPUs with no such memory type present will fail the allocation.

    Usage: Memory for transient attachment images (color attachments, depth attachments etc.), created with `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`.

    Allocations with this usage are always created as dedicated - it implies #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    */
    VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED = 6,
    /**
    Selects best memory type automatically.
    This flag is recommended for most common use cases.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO = 7,
    /**
    Selects best memory type automatically with preference for GPU (device) memory.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE = 8,
    /**
    Selects best memory type automatically with preference for CPU (host) memory.

    When using this flag, if you want to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT),
    you must pass one of the flags: #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    in VmaAllocationCreateInfo::flags.

    It can be used only with functions that let the library know `VkBufferCreateInfo` or `VkImageCreateInfo`, e.g.
    vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo()
    and not with generic memory allocation functions.
    */
    VMA_MEMORY_USAGE_AUTO_PREFER_HOST = 9,

    VMA_MEMORY_USAGE_MAX_ENUM = 0x7FFFFFFF
} VmaMemoryUsage;

/// Flags to be passed as VmaAllocationCreateInfo::flags.
typedef enum VmaAllocationCreateFlagBits
{
    /** \brief Set this flag if the allocation should have its own memory block.

    Use it for special, big resources, like fullscreen images used as attachments.
    */
    VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT = 0x00000001,

    /** \brief Set this flag to only try to allocate from existing `VkDeviceMemory` blocks and never create new such block.

    If new allocation cannot be placed in any of the existing blocks, allocation
    fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

    You should not use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT and
    #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT at the same time. It makes no sense.
    */
    VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT = 0x00000002,
    /** \brief Set this flag to use a memory that will be persistently mapped and retrieve pointer to it.

    Pointer to mapped memory will be returned through VmaAllocationInfo::pMappedData.

    It is valid to use this flag for allocation made from memory type that is not
    `HOST_VISIBLE`. This flag is then ignored and memory is not mapped. This is
    useful if you need an allocation that is efficient to use on GPU
    (`DEVICE_LOCAL`) and still want to map it directly if possible on platforms that
    support it (e.g. Intel GPU).
    */
    VMA_ALLOCATION_CREATE_MAPPED_BIT = 0x00000004,
    /** \deprecated Preserved for backward compatibility. Consider using vmaSetAllocationName() instead.

    Set this flag to treat VmaAllocationCreateInfo::pUserData as pointer to a
    null-terminated string. Instead of copying pointer value, a local copy of the
    string is made and stored in allocation's `pName`. The string is automatically
    freed together with the allocation. It is also used in vmaBuildStatsString().
    */
    VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT = 0x00000020,
    /** Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = 0x00000040,
    /** Create both buffer/image and allocation, but don't bind them together.
    It is useful when you want to bind yourself to do some more advanced binding, e.g. using some extensions.
    The flag is meaningful only with functions that bind by default: vmaCreateBuffer(), vmaCreateImage().
    Otherwise it is ignored.

    If you want to make sure the new buffer/image is not tied to the new memory allocation
    through `VkMemoryDedicatedAllocateInfoKHR` structure in case the allocation ends up in its own memory block,
    use also flag #VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT.
    */
    VMA_ALLOCATION_CREATE_DONT_BIND_BIT = 0x00000080,
    /** Create allocation only if additional device memory required for it, if any, won't exceed
    memory budget. Otherwise return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    */
    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT = 0x00000100,
    /** \brief Set this flag if the allocated memory will have aliasing resources.

    Usage of this flag prevents supplying `VkMemoryDedicatedAllocateInfoKHR` when #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT is specified.
    Otherwise created dedicated memory will not be suitable for aliasing resources, resulting in Vulkan Validation Layer errors.
    */
    VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT = 0x00000200,
    /**
    Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory will only be written sequentially, e.g. using `memcpy()` or a loop writing number-by-number,
    never read or accessed randomly, so a memory type can be selected that is uncached and write-combined.

    \warning Violating this declaration may work correctly, but will likely be very slow.
    Watch out for implicit reads introduced by doing e.g. `pMappedData[i] += x;`
    Better prepare your data in a local variable and `memcpy()` it to the mapped pointer all at once.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT = 0x00000400,
    /**
    Requests possibility to map the allocation (using vmaMapMemory() or #VMA_ALLOCATION_CREATE_MAPPED_BIT).

    - If you use #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` value,
      you must use this flag to be able to map the allocation. Otherwise, mapping is incorrect.
    - If you use other value of #VmaMemoryUsage, this flag is ignored and mapping is always possible in memory types that are `HOST_VISIBLE`.
      This includes allocations created in \ref custom_memory_pools.

    Declares that mapped memory can be read, written, and accessed in random order,
    so a `HOST_CACHED` memory type is required.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT = 0x00000800,
    /**
    Together with #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    it says that despite request for host access, a not-`HOST_VISIBLE` memory type can be selected
    if it may improve performance.

    By using this flag, you declare that you will check if the allocation ended up in a `HOST_VISIBLE` memory type
    (e.g. using vmaGetAllocationMemoryProperties()) and if not, you will create some "staging" buffer and
    issue an explicit transfer to write/read your data.
    To prepare for this possibility, don't forget to add appropriate flags like
    `VK_BUFFER_USAGE_TRANSFER_DST_BIT`, `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` to the parameters of created buffer or image.
    */
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT = 0x00001000,
    /** Allocation strategy that chooses smallest possible free range for the allocation
    to minimize memory usage and fragmentation, possibly at the expense of allocation time.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = 0x00010000,
    /** Allocation strategy that chooses first suitable free range for the allocation -
    not necessarily in terms of the smallest offset but the one that is easiest and fastest to find
    to minimize allocation time, possibly at the expense of allocation quality.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = 0x00020000,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    Used internally by defragmentation, not recommended in typical usage.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT  = 0x00040000,
    /** Alias to #VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** Alias to #VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_FIRST_FIT_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** A bit mask to extract only `STRATEGY` bits from entire set of flags.
    */
    VMA_ALLOCATION_CREATE_STRATEGY_MASK =
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT |
        VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,

    VMA_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaAllocationCreateFlagBits;
/// See #VmaAllocationCreateFlagBits.
typedef VkFlags VmaAllocationCreateFlags;

/// Flags to be passed as VmaPoolCreateInfo::flags.
typedef enum VmaPoolCreateFlagBits
{
    /** \brief Use this flag if you always allocate only buffers and linear images or only optimal images out of this pool and so Buffer-Image Granularity can be ignored.

    This is an optional optimization flag.

    If you always allocate using vmaCreateBuffer(), vmaCreateImage(),
    vmaAllocateMemoryForBuffer(), then you don't need to use it because allocator
    knows exact type of your allocations so it can handle Buffer-Image Granularity
    in the optimal way.

    If you also allocate using vmaAllocateMemoryForImage() or vmaAllocateMemory(),
    exact type of such allocations is not known, so allocator must be conservative
    in handling Buffer-Image Granularity, which can lead to suboptimal allocation
    (wasted memory). In that case, if you can make sure you always allocate only
    buffers and linear images or only optimal images out of this pool, use this flag
    to make allocator disregard Buffer-Image Granularity and so make allocations
    faster and more optimal.
    */
    VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT = 0x00000002,

    /** \brief Enables alternative, linear allocation algorithm in this pool.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT = 0x00000004,

    /** Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    VMA_POOL_CREATE_ALGORITHM_MASK =
        VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,

    VMA_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaPoolCreateFlagBits;
/// Flags to be passed as VmaPoolCreateInfo::flags. See #VmaPoolCreateFlagBits.
typedef VkFlags VmaPoolCreateFlags;

/// Flags to be passed as VmaDefragmentationInfo::flags.
typedef enum VmaDefragmentationFlagBits
{
    /* \brief Use simple but fast algorithm for defragmentation.
    May not achieve best results but will require least time to compute and least allocations to copy.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT = 0x1,
    /* \brief Default defragmentation algorithm, applied also when no `ALGORITHM` flag is specified.
    Offers a balance between defragmentation quality and the amount of allocations and bytes that need to be moved.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT = 0x2,
    /* \brief Perform full defragmentation of memory.
    Can result in notably more time to compute and allocations to copy, but will achieve best memory packing.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT = 0x4,
    /** \brief Use the most roboust algorithm at the cost of time to compute and number of copies to make.
    Only available when bufferImageGranularity is greater than 1, since it aims to reduce
    alignment issues between different types of resources.
    Otherwise falls back to same behavior as #VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT.
    */
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT = 0x8,

    /// A bit mask to extract only `ALGORITHM` bits from entire set of flags.
    VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK =
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT |
        VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT,

    VMA_DEFRAGMENTATION_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaDefragmentationFlagBits;
/// See #VmaDefragmentationFlagBits.
typedef VkFlags VmaDefragmentationFlags;

/// Operation performed on single defragmentation move. See structure #VmaDefragmentationMove.
typedef enum VmaDefragmentationMoveOperation
{
    /// Buffer/image has been recreated at `dstTmpAllocation`, data has been copied, old buffer/image has been destroyed. `srcAllocation` should be changed to point to the new place. This is the default value set by vmaBeginDefragmentationPass().
    VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY = 0,
    /// Set this value if you cannot move the allocation. New place reserved at `dstTmpAllocation` will be freed. `srcAllocation` will remain unchanged.
    VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE = 1,
    /// Set this value if you decide to abandon the allocation and you destroyed the buffer/image. New place reserved at `dstTmpAllocation` will be freed, along with `srcAllocation`, which will be destroyed.
    VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY = 2,
} VmaDefragmentationMoveOperation;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Flags to be passed as VmaVirtualBlockCreateInfo::flags.
typedef enum VmaVirtualBlockCreateFlagBits
{
    /** \brief Enables alternative, linear allocation algorithm in this virtual block.

    Specify this flag to enable linear allocation algorithm, which always creates
    new allocations after last one and doesn't reuse space from allocations freed in
    between. It trades memory consumption for simplified algorithm and data
    structure, which has better performance and uses less memory for metadata.

    By using this flag, you can achieve behavior of free-at-once, stack,
    ring buffer, and double stack.
    For details, see documentation chapter \ref linear_algorithm.
    */
    VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT = 0x00000001,

    /** \brief Bit mask to extract only `ALGORITHM` bits from entire set of flags.
    */
    VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK =
        VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT,

    VMA_VIRTUAL_BLOCK_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaVirtualBlockCreateFlagBits;
/// Flags to be passed as VmaVirtualBlockCreateInfo::flags. See #VmaVirtualBlockCreateFlagBits.
typedef VkFlags VmaVirtualBlockCreateFlags;

/// Flags to be passed as VmaVirtualAllocationCreateInfo::flags.
typedef enum VmaVirtualAllocationCreateFlagBits
{
    /** \brief Allocation will be created from upper stack in a double stack pool.

    This flag is only allowed for virtual blocks created with #VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT flag.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT = VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT,
    /** \brief Allocation strategy that tries to minimize memory usage.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT,
    /** \brief Allocation strategy that tries to minimize allocation time.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
    /** Allocation strategy that chooses always the lowest offset in available space.
    This is not the most efficient strategy but achieves highly packed data.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT = VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
    /** \brief A bit mask to extract only `STRATEGY` bits from entire set of flags.

    These strategy flags are binary compatible with equivalent flags in #VmaAllocationCreateFlagBits.
    */
    VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK = VMA_ALLOCATION_CREATE_STRATEGY_MASK,

    VMA_VIRTUAL_ALLOCATION_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VmaVirtualAllocationCreateFlagBits;
/// Flags to be passed as VmaVirtualAllocationCreateInfo::flags. See #VmaVirtualAllocationCreateFlagBits.
typedef VkFlags VmaVirtualAllocationCreateFlags;

/** @} */

#endif // _VMA_ENUM_DECLARATIONS

#ifndef _VMA_DATA_TYPES_DECLARATIONS

/**
\addtogroup group_init
@{ */

/** \struct VmaAllocator
\brief Represents main object of this library initialized.

Fill structure #VmaAllocatorCreateInfo and call function vmaCreateAllocator() to create it.
Call function vmaDestroyAllocator() to destroy it.

It is recommended to create just one object of this type per `VkDevice` object,
right after Vulkan is initialized and keep it alive until before Vulkan device is destroyed.
*/
VK_DEFINE_HANDLE(VmaAllocator)

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \struct VmaPool
\brief Represents custom memory pool

Fill structure VmaPoolCreateInfo and call function vmaCreatePool() to create it.
Call function vmaDestroyPool() to destroy it.

For more information see [Custom memory pools](@ref choosing_memory_type_custom_memory_pools).
*/
VK_DEFINE_HANDLE(VmaPool)

/** \struct VmaAllocation
\brief Represents single memory allocation.

It may be either dedicated block of `VkDeviceMemory` or a specific region of a bigger block of this type
plus unique offset.

There are multiple ways to create such object.
You need to fill structure VmaAllocationCreateInfo.
For more information see [Choosing memory type](@ref choosing_memory_type).

Although the library provides convenience functions that create Vulkan buffer or image,
allocate memory for it and bind them together,
binding of the allocation to a buffer or an image is out of scope of the allocation itself.
Allocation object can exist without buffer/image bound,
binding can be done manually by the user, and destruction of it can be done
independently of destruction of the allocation.

The object also remembers its size and some other information.
To retrieve this information, use function vmaGetAllocationInfo() and inspect
returned structure VmaAllocationInfo.
*/
VK_DEFINE_HANDLE(VmaAllocation)

/** \struct VmaDefragmentationContext
\brief An opaque object that represents started defragmentation process.

Fill structure #VmaDefragmentationInfo and call function vmaBeginDefragmentation() to create it.
Call function vmaEndDefragmentation() to destroy it.
*/
VK_DEFINE_HANDLE(VmaDefragmentationContext)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct VmaVirtualAllocation
\brief Represents single memory allocation done inside VmaVirtualBlock.

Use it as a unique identifier to virtual allocation within the single block.

Use value `VK_NULL_HANDLE` to represent a null/invalid allocation.
*/
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VmaVirtualAllocation)

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \struct VmaVirtualBlock
\brief Handle to a virtual block object that allows to use core allocation algorithm without allocating any real GPU memory.

Fill in #VmaVirtualBlockCreateInfo structure and use vmaCreateVirtualBlock() to create it. Use vmaDestroyVirtualBlock() to destroy it.
For more information, see documentation chapter \ref virtual_allocator.

This object is not thread-safe - should not be used from multiple threads simultaneously, must be synchronized externally.
*/
VK_DEFINE_HANDLE(VmaVirtualBlock)

/** @} */

/**
\addtogroup group_init
@{
*/

/// Callback function called after successful vkAllocateMemory.
typedef void (VKAPI_PTR* PFN_vmaAllocateDeviceMemoryFunction)(
    VmaAllocator VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* VMA_NULLABLE                           pUserData);

/// Callback function called before vkFreeMemory.
typedef void (VKAPI_PTR* PFN_vmaFreeDeviceMemoryFunction)(
    VmaAllocator VMA_NOT_NULL                    allocator,
    uint32_t                                     memoryType,
    VkDeviceMemory VMA_NOT_NULL_NON_DISPATCHABLE memory,
    VkDeviceSize                                 size,
    void* VMA_NULLABLE                           pUserData);

/** \brief Set of callbacks that the library will call for `vkAllocateMemory` and `vkFreeMemory`.

Provided for informative purpose, e.g. to gather statistics about number of
allocations or total amount of memory allocated in Vulkan.

Used in VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.
*/
typedef struct VmaDeviceMemoryCallbacks
{
    /// Optional, can be null.
    PFN_vmaAllocateDeviceMemoryFunction VMA_NULLABLE pfnAllocate;
    /// Optional, can be null.
    PFN_vmaFreeDeviceMemoryFunction VMA_NULLABLE pfnFree;
    /// Optional, can be null.
    void* VMA_NULLABLE pUserData;
} VmaDeviceMemoryCallbacks;

/** \brief Pointers to some Vulkan functions - a subset used by the library.

Used in VmaAllocatorCreateInfo::pVulkanFunctions.
*/
typedef struct VmaVulkanFunctions
{
    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetInstanceProcAddr VMA_NULLABLE vkGetInstanceProcAddr;
    /// Required when using VMA_DYNAMIC_VULKAN_FUNCTIONS.
    PFN_vkGetDeviceProcAddr VMA_NULLABLE vkGetDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties VMA_NULLABLE vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties;
    PFN_vkAllocateMemory VMA_NULLABLE vkAllocateMemory;
    PFN_vkFreeMemory VMA_NULLABLE vkFreeMemory;
    PFN_vkMapMemory VMA_NULLABLE vkMapMemory;
    PFN_vkUnmapMemory VMA_NULLABLE vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges VMA_NULLABLE vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges VMA_NULLABLE vkInvalidateMappedMemoryRanges;
    PFN_vkBindBufferMemory VMA_NULLABLE vkBindBufferMemory;
    PFN_vkBindImageMemory VMA_NULLABLE vkBindImageMemory;
    PFN_vkGetBufferMemoryRequirements VMA_NULLABLE vkGetBufferMemoryRequirements;
    PFN_vkGetImageMemoryRequirements VMA_NULLABLE vkGetImageMemoryRequirements;
    PFN_vkCreateBuffer VMA_NULLABLE vkCreateBuffer;
    PFN_vkDestroyBuffer VMA_NULLABLE vkDestroyBuffer;
    PFN_vkCreateImage VMA_NULLABLE vkCreateImage;
    PFN_vkDestroyImage VMA_NULLABLE vkDestroyImage;
    PFN_vkCmdCopyBuffer VMA_NULLABLE vkCmdCopyBuffer;
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkGetBufferMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetBufferMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetBufferMemoryRequirements2KHR VMA_NULLABLE vkGetBufferMemoryRequirements2KHR;
    /// Fetch "vkGetImageMemoryRequirements2" on Vulkan >= 1.1, fetch "vkGetImageMemoryRequirements2KHR" when using VK_KHR_dedicated_allocation extension.
    PFN_vkGetImageMemoryRequirements2KHR VMA_NULLABLE vkGetImageMemoryRequirements2KHR;
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    /// Fetch "vkBindBufferMemory2" on Vulkan >= 1.1, fetch "vkBindBufferMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindBufferMemory2KHR VMA_NULLABLE vkBindBufferMemory2KHR;
    /// Fetch "vkBindImageMemory2" on Vulkan >= 1.1, fetch "vkBindImageMemory2KHR" when using VK_KHR_bind_memory2 extension.
    PFN_vkBindImageMemory2KHR VMA_NULLABLE vkBindImageMemory2KHR;
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR VMA_NULLABLE vkGetPhysicalDeviceMemoryProperties2KHR;
#endif
#if VMA_VULKAN_VERSION >= 1003000
    /// Fetch from "vkGetDeviceBufferMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceBufferMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceBufferMemoryRequirements VMA_NULLABLE vkGetDeviceBufferMemoryRequirements;
    /// Fetch from "vkGetDeviceImageMemoryRequirements" on Vulkan >= 1.3, but you can also fetch it from "vkGetDeviceImageMemoryRequirementsKHR" if you enabled extension VK_KHR_maintenance4.
    PFN_vkGetDeviceImageMemoryRequirements VMA_NULLABLE vkGetDeviceImageMemoryRequirements;
#endif
} VmaVulkanFunctions;

/// Description of a Allocator to be created.
typedef struct VmaAllocatorCreateInfo
{
    /// Flags for created allocator. Use #VmaAllocatorCreateFlagBits enum.
    VmaAllocatorCreateFlags flags;
    /// Vulkan physical device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkPhysicalDevice VMA_NOT_NULL physicalDevice;
    /// Vulkan device.
    /** It must be valid throughout whole lifetime of created allocator. */
    VkDevice VMA_NOT_NULL device;
    /// Preferred size of a single `VkDeviceMemory` block to be allocated from large heaps > 1 GiB. Optional.
    /** Set to 0 to use default, which is currently 256 MiB. */
    VkDeviceSize preferredLargeHeapBlockSize;
    /// Custom CPU memory allocation callbacks. Optional.
    /** Optional, can be null. When specified, will also be used for all CPU-side memory allocations. */
    const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks;
    /// Informative callbacks for `vkAllocateMemory`, `vkFreeMemory`. Optional.
    /** Optional, can be null. */
    const VmaDeviceMemoryCallbacks* VMA_NULLABLE pDeviceMemoryCallbacks;
    /** \brief Either null or a pointer to an array of limits on maximum number of bytes that can be allocated out of particular Vulkan memory heap.

    If not NULL, it must be a pointer to an array of
    `VkPhysicalDeviceMemoryProperties::memoryHeapCount` elements, defining limit on
    maximum number of bytes that can be allocated out of particular Vulkan memory
    heap.

    Any of the elements may be equal to `VK_WHOLE_SIZE`, which means no limit on that
    heap. This is also the default in case of `pHeapSizeLimit` = NULL.

    If there is a limit defined for a heap:

    - If user tries to allocate more memory from that heap using this allocator,
      the allocation fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
    - If the limit is smaller than heap size reported in `VkMemoryHeap::size`, the
      value of this limit will be reported instead when using vmaGetMemoryProperties().

    Warning! Using this feature may not be equivalent to installing a GPU with
    smaller amount of memory, because graphics driver doesn't necessary fail new
    allocations with `VK_ERROR_OUT_OF_DEVICE_MEMORY` result when memory capacity is
    exceeded. It may return success and just silently migrate some device memory
    blocks to system RAM. This driver behavior can also be controlled using
    VK_AMD_memory_overallocation_behavior extension.
    */
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pHeapSizeLimit;

    /** \brief Pointers to Vulkan functions. Can be null.

    For details see [Pointers to Vulkan functions](@ref config_Vulkan_functions).
    */
    const VmaVulkanFunctions* VMA_NULLABLE pVulkanFunctions;
    /** \brief Handle to Vulkan instance object.

    Starting from version 3.0.0 this member is no longer optional, it must be set!
    */
    VkInstance VMA_NOT_NULL instance;
    /** \brief Optional. The highest version of Vulkan that the application is designed to use.

    It must be a value in the format as created by macro `VK_MAKE_VERSION` or a constant like: `VK_API_VERSION_1_1`, `VK_API_VERSION_1_0`.
    The patch version number specified is ignored. Only the major and minor versions are considered.
    It must be less or equal (preferably equal) to value as passed to `vkCreateInstance` as `VkApplicationInfo::apiVersion`.
    Only versions 1.0, 1.1, 1.2, 1.3 are supported by the current implementation.
    Leaving it initialized to zero is equivalent to `VK_API_VERSION_1_0`.
    */
    uint32_t vulkanApiVersion;
#if VMA_EXTERNAL_MEMORY
    /** \brief Either null or a pointer to an array of external memory handle types for each Vulkan memory type.

    If not NULL, it must be a pointer to an array of `VkPhysicalDeviceMemoryProperties::memoryTypeCount`
    elements, defining external memory handle types of particular Vulkan memory type,
    to be passed using `VkExportMemoryAllocateInfoKHR`.

    Any of the elements may be equal to 0, which means not to use `VkExportMemoryAllocateInfoKHR` on this memory type.
    This is also the default in case of `pTypeExternalMemoryHandleTypes` = NULL.
    */
    const VkExternalMemoryHandleTypeFlagsKHR* VMA_NULLABLE VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryTypeCount") pTypeExternalMemoryHandleTypes;
#endif // #if VMA_EXTERNAL_MEMORY
} VmaAllocatorCreateInfo;

/// Information about existing #VmaAllocator object.
typedef struct VmaAllocatorInfo
{
    /** \brief Handle to Vulkan instance object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::instance.
    */
    VkInstance VMA_NOT_NULL instance;
    /** \brief Handle to Vulkan physical device object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::physicalDevice.
    */
    VkPhysicalDevice VMA_NOT_NULL physicalDevice;
    /** \brief Handle to Vulkan device object.

    This is the same value as has been passed through VmaAllocatorCreateInfo::device.
    */
    VkDevice VMA_NOT_NULL device;
} VmaAllocatorInfo;

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Calculated statistics of memory usage e.g. in a specific memory type, heap, custom pool, or total.

These are fast to calculate.
See functions: vmaGetHeapBudgets(), vmaGetPoolStatistics().
*/
typedef struct VmaStatistics
{
    /** \brief Number of `VkDeviceMemory` objects - Vulkan memory blocks allocated.
    */
    uint32_t blockCount;
    /** \brief Number of #VmaAllocation objects allocated.

    Dedicated allocations have their own blocks, so each one adds 1 to `allocationCount` as well as `blockCount`.
    */
    uint32_t allocationCount;
    /** \brief Number of bytes allocated in `VkDeviceMemory` blocks.

    \note To avoid confusion, please be aware that what Vulkan calls an "allocation" - a whole `VkDeviceMemory` object
    (e.g. as in `VkPhysicalDeviceLimits::maxMemoryAllocationCount`) is called a "block" in VMA, while VMA calls
    "allocation" a #VmaAllocation object that represents a memory region sub-allocated from such block, usually for a single buffer or image.
    */
    VkDeviceSize blockBytes;
    /** \brief Total number of bytes occupied by all #VmaAllocation objects.

    Always less or equal than `blockBytes`.
    Difference `(blockBytes - allocationBytes)` is the amount of memory allocated from Vulkan
    but unused by any #VmaAllocation.
    */
    VkDeviceSize allocationBytes;
} VmaStatistics;

/** \brief More detailed statistics than #VmaStatistics.

These are slower to calculate. Use for debugging purposes.
See functions: vmaCalculateStatistics(), vmaCalculatePoolStatistics().

Previous version of the statistics API provided averages, but they have been removed
because they can be easily calculated as:

\code
VkDeviceSize allocationSizeAvg = detailedStats.statistics.allocationBytes / detailedStats.statistics.allocationCount;
VkDeviceSize unusedBytes = detailedStats.statistics.blockBytes - detailedStats.statistics.allocationBytes;
VkDeviceSize unusedRangeSizeAvg = unusedBytes / detailedStats.unusedRangeCount;
\endcode
*/
typedef struct VmaDetailedStatistics
{
    /// Basic statistics.
    VmaStatistics statistics;
    /// Number of free ranges of memory between allocations.
    uint32_t unusedRangeCount;
    /// Smallest allocation size. `VK_WHOLE_SIZE` if there are 0 allocations.
    VkDeviceSize allocationSizeMin;
    /// Largest allocation size. 0 if there are 0 allocations.
    VkDeviceSize allocationSizeMax;
    /// Smallest empty range size. `VK_WHOLE_SIZE` if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMin;
    /// Largest empty range size. 0 if there are 0 empty ranges.
    VkDeviceSize unusedRangeSizeMax;
} VmaDetailedStatistics;

/** \brief  General statistics from current state of the Allocator -
total memory usage across all memory heaps and types.

These are slower to calculate. Use for debugging purposes.
See function vmaCalculateStatistics().
*/
typedef struct VmaTotalStatistics
{
    VmaDetailedStatistics memoryType[VK_MAX_MEMORY_TYPES];
    VmaDetailedStatistics memoryHeap[VK_MAX_MEMORY_HEAPS];
    VmaDetailedStatistics total;
} VmaTotalStatistics;

/** \brief Statistics of current memory usage and available budget for a specific memory heap.

These are fast to calculate.
See function vmaGetHeapBudgets().
*/
typedef struct VmaBudget
{
    /** \brief Statistics fetched from the library.
    */
    VmaStatistics statistics;
    /** \brief Estimated current memory usage of the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different than `statistics.blockBytes` (usually higher) due to additional implicit objects
    also occupying the memory, like swapchain, pipelines, descriptor heaps, command buffers, or
    `VkDeviceMemory` blocks allocated outside of this library, if any.
    */
    VkDeviceSize usage;
    /** \brief Estimated amount of memory available to the program, in bytes.

    Fetched from system using VK_EXT_memory_budget extension if enabled.

    It might be different (most probably smaller) than `VkMemoryHeap::size[heapIndex]` due to factors
    external to the program, decided by the operating system.
    Difference `budget - usage` is the amount of additional memory that can probably
    be allocated without problems. Exceeding the budget may result in various problems.
    */
    VkDeviceSize budget;
} VmaBudget;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Parameters of new #VmaAllocation.

To be used with functions like vmaCreateBuffer(), vmaCreateImage(), and many others.
*/
typedef struct VmaAllocationCreateInfo
{
    /// Use #VmaAllocationCreateFlagBits enum.
    VmaAllocationCreateFlags flags;
    /** \brief Intended usage of memory.

    You can leave #VMA_MEMORY_USAGE_UNKNOWN if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.
    */
    VmaMemoryUsage usage;
    /** \brief Flags that must be set in a Memory Type chosen for an allocation.

    Leave 0 if you specify memory requirements in other way. \n
    If `pool` is not null, this member is ignored.*/
    VkMemoryPropertyFlags requiredFlags;
    /** \brief Flags that preferably should be set in a memory type chosen for an allocation.

    Set to 0 if no additional flags are preferred. \n
    If `pool` is not null, this member is ignored. */
    VkMemoryPropertyFlags preferredFlags;
    /** \brief Bitmask containing one bit set for every memory type acceptable for this allocation.

    Value 0 is equivalent to `UINT32_MAX` - it means any memory type is accepted if
    it meets other requirements specified by this structure, with no further
    restrictions on memory type index. \n
    If `pool` is not null, this member is ignored.
    */
    uint32_t memoryTypeBits;
    /** \brief Pool that this allocation should be created in.

    Leave `VK_NULL_HANDLE` to allocate from default pool. If not null, members:
    `usage`, `requiredFlags`, `preferredFlags`, `memoryTypeBits` are ignored.
    */
    VmaPool VMA_NULLABLE pool;
    /** \brief Custom general-purpose pointer that will be stored in #VmaAllocation, can be read as VmaAllocationInfo::pUserData and changed using vmaSetAllocationUserData().

    If #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT is used, it must be either
    null or pointer to a null-terminated string. The string will be then copied to
    internal buffer, so it doesn't need to be valid after allocation call.
    */
    void* VMA_NULLABLE pUserData;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocation relative to other memory allocations.

    It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object
    and this allocation ends up as dedicated or is explicitly forced as dedicated using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
    Otherwise, it has the priority of a memory block where it is placed and this variable is ignored.
    */
    float priority;
} VmaAllocationCreateInfo;

/// Describes parameter of created #VmaPool.
typedef struct VmaPoolCreateInfo
{
    /** \brief Vulkan memory type index to allocate this pool from.
    */
    uint32_t memoryTypeIndex;
    /** \brief Use combination of #VmaPoolCreateFlagBits.
    */
    VmaPoolCreateFlags flags;
    /** \brief Size of a single `VkDeviceMemory` block to be allocated as part of this pool, in bytes. Optional.

    Specify nonzero to set explicit, constant size of memory blocks used by this
    pool.

    Leave 0 to use default and let the library manage block sizes automatically.
    Sizes of particular blocks may vary.
    In this case, the pool will also support dedicated allocations.
    */
    VkDeviceSize blockSize;
    /** \brief Minimum number of blocks to be always allocated in this pool, even if they stay empty.

    Set to 0 to have no preallocated blocks and allow the pool be completely empty.
    */
    size_t minBlockCount;
    /** \brief Maximum number of blocks that can be allocated in this pool. Optional.

    Set to 0 to use default, which is `SIZE_MAX`, which means no limit.

    Set to same value as VmaPoolCreateInfo::minBlockCount to have fixed amount of memory allocated
    throughout whole lifetime of this pool.
    */
    size_t maxBlockCount;
    /** \brief A floating-point value between 0 and 1, indicating the priority of the allocations in this pool relative to other memory allocations.

    It is used only when #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT flag was used during creation of the #VmaAllocator object.
    Otherwise, this variable is ignored.
    */
    float priority;
    /** \brief Additional minimum alignment to be used for all allocations created from this pool. Can be 0.

    Leave 0 (default) not to impose any additional alignment. If not 0, it must be a power of two.
    It can be useful in cases where alignment returned by Vulkan by functions like `vkGetBufferMemoryRequirements` is not enough,
    e.g. when doing interop with OpenGL.
    */
    VkDeviceSize minAllocationAlignment;
    /** \brief Additional `pNext` chain to be attached to `VkMemoryAllocateInfo` used for every allocation made by this pool. Optional.

    Optional, can be null. If not null, it must point to a `pNext` chain of structures that can be attached to `VkMemoryAllocateInfo`.
    It can be useful for special needs such as adding `VkExportMemoryAllocateInfoKHR`.
    Structures pointed by this member must remain alive and unchanged for the whole lifetime of the custom pool.

    Please note that some structures, e.g. `VkMemoryPriorityAllocateInfoEXT`, `VkMemoryDedicatedAllocateInfoKHR`,
    can be attached automatically by this library when using other, more convenient of its features.
    */
    void* VMA_NULLABLE pMemoryAllocateNext;
} VmaPoolCreateInfo;

/** @} */

/**
\addtogroup group_alloc
@{
*/

/// Parameters of #VmaAllocation objects, that can be retrieved using function vmaGetAllocationInfo().
typedef struct VmaAllocationInfo
{
    /** \brief Memory type index that this allocation was allocated from.

    It never changes.
    */
    uint32_t memoryType;
    /** \brief Handle to Vulkan memory object.

    Same memory object can be shared by multiple allocations.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceMemory VMA_NULLABLE_NON_DISPATCHABLE deviceMemory;
    /** \brief Offset in `VkDeviceMemory` object to the beginning of this allocation, in bytes. `(deviceMemory, offset)` pair is unique to this allocation.

    You usually don't need to use this offset. If you create a buffer or an image together with the allocation using e.g. function
    vmaCreateBuffer(), vmaCreateImage(), functions that operate on these resources refer to the beginning of the buffer or image,
    not entire device memory block. Functions like vmaMapMemory(), vmaBindBufferMemory() also refer to the beginning of the allocation
    and apply this offset automatically.

    It can change after the allocation is moved during \ref defragmentation.
    */
    VkDeviceSize offset;
    /** \brief Size of this allocation, in bytes.

    It never changes.

    \note Allocation size returned in this variable may be greater than the size
    requested for the resource e.g. as `VkBufferCreateInfo::size`. Whole size of the
    allocation is accessible for operations on memory e.g. using a pointer after
    mapping with vmaMapMemory(), but operations on the resource e.g. using
    `vkCmdCopyBuffer` must be limited to the size of the resource.
    */
    VkDeviceSize size;
    /** \brief Pointer to the beginning of this allocation as mapped data.

    If the allocation hasn't been mapped using vmaMapMemory() and hasn't been
    created with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag, this value is null.

    It can change after call to vmaMapMemory(), vmaUnmapMemory().
    It can also change after the allocation is moved during \ref defragmentation.
    */
    void* VMA_NULLABLE pMappedData;
    /** \brief Custom general-purpose pointer that was passed as VmaAllocationCreateInfo::pUserData or set using vmaSetAllocationUserData().

    It can change after call to vmaSetAllocationUserData() for this allocation.
    */
    void* VMA_NULLABLE pUserData;
    /** \brief Custom allocation name that was set with vmaSetAllocationName().

    It can change after call to vmaSetAllocationName() for this allocation.

    Another way to set custom name is to pass it in VmaAllocationCreateInfo::pUserData with
    additional flag #VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT set [DEPRECATED].
    */
    const char* VMA_NULLABLE pName;
} VmaAllocationInfo;

/** \brief Parameters for defragmentation.

To be used with function vmaBeginDefragmentation().
*/
typedef struct VmaDefragmentationInfo
{
    /// \brief Use combination of #VmaDefragmentationFlagBits.
    VmaDefragmentationFlags flags;
    /** \brief Custom pool to be defragmented.

    If null then default pools will undergo defragmentation process.
    */
    VmaPool VMA_NULLABLE pool;
    /** \brief Maximum numbers of bytes that can be copied during single pass, while moving allocations to different places.

    `0` means no limit.
    */
    VkDeviceSize maxBytesPerPass;
    /** \brief Maximum number of allocations that can be moved during single pass to a different place.

    `0` means no limit.
    */
    uint32_t maxAllocationsPerPass;
} VmaDefragmentationInfo;

/// Single move of an allocation to be done for defragmentation.
typedef struct VmaDefragmentationMove
{
    /// Operation to be performed on the allocation by vmaEndDefragmentationPass(). Default value is #VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY. You can modify it.
    VmaDefragmentationMoveOperation operation;
    /// Allocation that should be moved.
    VmaAllocation VMA_NOT_NULL srcAllocation;
    /** \brief Temporary allocation pointing to destination memory that will replace `srcAllocation`.

    \warning Do not store this allocation in your data structures! It exists only temporarily, for the duration of the defragmentation pass,
    to be used for binding new buffer/image to the destination memory using e.g. vmaBindBufferMemory().
    vmaEndDefragmentationPass() will destroy it and make `srcAllocation` point to this memory.
    */
    VmaAllocation VMA_NOT_NULL dstTmpAllocation;
} VmaDefragmentationMove;

/** \brief Parameters for incremental defragmentation steps.

To be used with function vmaBeginDefragmentationPass().
*/
typedef struct VmaDefragmentationPassMoveInfo
{
    /// Number of elements in the `pMoves` array.
    uint32_t moveCount;
    /** \brief Array of moves to be performed by the user in the current defragmentation pass.

    Pointer to an array of `moveCount` elements, owned by VMA, created in vmaBeginDefragmentationPass(), destroyed in vmaEndDefragmentationPass().

    For each element, you should:

    1. Create a new buffer/image in the place pointed by VmaDefragmentationMove::dstMemory + VmaDefragmentationMove::dstOffset.
    2. Copy data from the VmaDefragmentationMove::srcAllocation e.g. using `vkCmdCopyBuffer`, `vkCmdCopyImage`.
    3. Make sure these commands finished executing on the GPU.
    4. Destroy the old buffer/image.

    Only then you can finish defragmentation pass by calling vmaEndDefragmentationPass().
    After this call, the allocation will point to the new place in memory.

    Alternatively, if you cannot move specific allocation, you can set VmaDefragmentationMove::operation to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.

    Alternatively, if you decide you want to completely remove the allocation:

    1. Destroy its buffer/image.
    2. Set VmaDefragmentationMove::operation to #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.

    Then, after vmaEndDefragmentationPass() the allocation will be freed.
    */
    VmaDefragmentationMove* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(moveCount) pMoves;
} VmaDefragmentationPassMoveInfo;

/// Statistics returned for defragmentation process in function vmaEndDefragmentation().
typedef struct VmaDefragmentationStats
{
    /// Total number of bytes that have been copied while moving allocations to different places.
    VkDeviceSize bytesMoved;
    /// Total number of bytes that have been released to the system by freeing empty `VkDeviceMemory` objects.
    VkDeviceSize bytesFreed;
    /// Number of allocations that have been moved to different places.
    uint32_t allocationsMoved;
    /// Number of empty `VkDeviceMemory` objects that have been released to the system.
    uint32_t deviceMemoryBlocksFreed;
} VmaDefragmentationStats;

/** @} */

/**
\addtogroup group_virtual
@{
*/

/// Parameters of created #VmaVirtualBlock object to be passed to vmaCreateVirtualBlock().
typedef struct VmaVirtualBlockCreateInfo
{
    /** \brief Total size of the virtual block.

    Sizes can be expressed in bytes or any units you want as long as you are consistent in using them.
    For example, if you allocate from some array of structures, 1 can mean single instance of entire structure.
    */
    VkDeviceSize size;

    /** \brief Use combination of #VmaVirtualBlockCreateFlagBits.
    */
    VmaVirtualBlockCreateFlags flags;

    /** \brief Custom CPU memory allocation callbacks. Optional.

    Optional, can be null. When specified, they will be used for all CPU-side memory allocations.
    */
    const VkAllocationCallbacks* VMA_NULLABLE pAllocationCallbacks;
} VmaVirtualBlockCreateInfo;

/// Parameters of created virtual allocation to be passed to vmaVirtualAllocate().
typedef struct VmaVirtualAllocationCreateInfo
{
    /** \brief Size of the allocation.

    Cannot be zero.
    */
    VkDeviceSize size;
    /** \brief Required alignment of the allocation. Optional.

    Must be power of two. Special value 0 has the same meaning as 1 - means no special alignment is required, so allocation can start at any offset.
    */
    VkDeviceSize alignment;
    /** \brief Use combination of #VmaVirtualAllocationCreateFlagBits.
    */
    VmaVirtualAllocationCreateFlags flags;
    /** \brief Custom pointer to be associated with the allocation. Optional.

    It can be any value and can be used for user-defined purposes. It can be fetched or changed later.
    */
    void* VMA_NULLABLE pUserData;
} VmaVirtualAllocationCreateInfo;

/// Parameters of an existing virtual allocation, returned by vmaGetVirtualAllocationInfo().
typedef struct VmaVirtualAllocationInfo
{
    /** \brief Offset of the allocation.

    Offset at which the allocation was made.
    */
    VkDeviceSize offset;
    /** \brief Size of the allocation.

    Same value as passed in VmaVirtualAllocationCreateInfo::size.
    */
    VkDeviceSize size;
    /** \brief Custom pointer associated with the allocation.

    Same value as passed in VmaVirtualAllocationCreateInfo::pUserData or to vmaSetVirtualAllocationUserData().
    */
    void* VMA_NULLABLE pUserData;
} VmaVirtualAllocationInfo;

/** @} */

#endif // _VMA_DATA_TYPES_DECLARATIONS

#ifndef _VMA_FUNCTION_HEADERS

/**
\addtogroup group_init
@{
*/

/// Creates #VmaAllocator object.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocator VMA_NULLABLE* VMA_NOT_NULL pAllocator);

/// Destroys allocator object.
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator VMA_NULLABLE allocator);

/** \brief Returns information about existing #VmaAllocator object - handle to Vulkan device etc.

It might be useful if you want to keep just the #VmaAllocator handle and fetch other required handles to
`VkPhysicalDevice`, `VkDevice` etc. every time using this function.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocatorInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocatorInfo* VMA_NOT_NULL pAllocatorInfo);

/**
PhysicalDeviceProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkPhysicalDeviceProperties* VMA_NULLABLE* VMA_NOT_NULL ppPhysicalDeviceProperties);

/**
PhysicalDeviceMemoryProperties are fetched from physicalDevice by the allocator.
You can access it here, without fetching it again on your own.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkPhysicalDeviceMemoryProperties* VMA_NULLABLE* VMA_NOT_NULL ppPhysicalDeviceMemoryProperties);

/**
\brief Given Memory Type Index, returns Property Flags of this memory type.

This is just a convenience function. Same information can be obtained using
vmaGetMemoryProperties().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags);

/** \brief Sets index of the current frame.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t frameIndex);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics from current state of the Allocator.

This function is called "calculate" not "get" because it has to traverse all
internal data structures, so it may be quite slow. Use it for debugging purposes.
For faster but more brief statistics suitable to be called every frame or every allocation,
use vmaGetHeapBudgets().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaTotalStatistics* VMA_NOT_NULL pStats);

/** \brief Retrieves information about current memory usage and budget for all memory heaps.

\param allocator
\param[out] pBudgets Must point to array with number of elements at least equal to number of memory heaps in physical device used.

This function is called "get" not "calculate" because it is very fast, suitable to be called
every frame or every allocation. For more detailed statistics use vmaCalculateStatistics().

Note that when using allocator from multiple threads, returned information may immediately
become outdated.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetHeapBudgets(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaBudget* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL("VkPhysicalDeviceMemoryProperties::memoryHeapCount") pBudgets);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/**
\brief Helps to find memoryTypeIndex, given memoryTypeBits and VmaAllocationCreateInfo.

This algorithm tries to find a memory type that:

- Is allowed by memoryTypeBits.
- Contains all the flags from pAllocationCreateInfo->requiredFlags.
- Matches intended usage.
- Has as many flags from pAllocationCreateInfo->preferredFlags as possible.

\return Returns VK_ERROR_FEATURE_NOT_PRESENT if not found. Receiving such result
from this function or any other allocating function probably means that your
device doesn't support any memory type with requested features for the specific
type of resource you want to use it for. Please check parameters of your
resource, like image layout (OPTIMAL versus LINEAR) or mip level count.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find memoryTypeIndex, given VkBufferCreateInfo and VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It internally creates a temporary, dummy buffer that never has memory bound.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/**
\brief Helps to find memoryTypeIndex, given VkImageCreateInfo and VmaAllocationCreateInfo.

It can be useful e.g. to determine value to be used as VmaPoolCreateInfo::memoryTypeIndex.
It internally creates a temporary, dummy image that never has memory bound.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    uint32_t* VMA_NOT_NULL pMemoryTypeIndex);

/** \brief Allocates Vulkan device memory and creates #VmaPool object.

\param allocator Allocator object.
\param pCreateInfo Parameters of pool to create.
\param[out] pPool Handle to created pool.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
    VmaAllocator VMA_NOT_NULL allocator,
    const VmaPoolCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaPool VMA_NULLABLE* VMA_NOT_NULL pPool);

/** \brief Destroys #VmaPool object and frees Vulkan device memory.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NULLABLE pool);

/** @} */

/**
\addtogroup group_stats
@{
*/

/** \brief Retrieves statistics of existing #VmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    VmaStatistics* VMA_NOT_NULL pPoolStats);

/** \brief Retrieves detailed statistics of existing #VmaPool object.

\param allocator Allocator object.
\param pool Pool object.
\param[out] pPoolStats Statistics of specified pool.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculatePoolStatistics(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    VmaDetailedStatistics* VMA_NOT_NULL pPoolStats);

/** @} */

/**
\addtogroup group_alloc
@{
*/

/** \brief Checks magic number in margins around all allocations in given memory pool in search for corruptions.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and the pool is created in memory type that is
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for specified pool.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool);

/** \brief Retrieves name of a custom pool.

After the call `ppName` is either null or points to an internally-owned null-terminated string
containing name of the pool that was previously set. The pointer becomes invalid when the pool is
destroyed or its name is changed using vmaSetPoolName().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    const char* VMA_NULLABLE* VMA_NOT_NULL ppName);

/** \brief Sets name of a custom pool.

`pName` can be either null or pointer to a null-terminated string with new name for the pool.
Function makes internal copy of the string, so it can be changed or freed immediately after this call.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaPool VMA_NOT_NULL pool,
    const char* VMA_NULLABLE pName);

/** \brief General purpose memory allocation.

\param allocator
\param pVkMemoryRequirements
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

It is recommended to use vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage(),
vmaCreateBuffer(), vmaCreateImage() instead whenever possible.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkMemoryRequirements* VMA_NOT_NULL pVkMemoryRequirements,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief General purpose memory allocation for multiple allocation objects at once.

\param allocator Allocator object.
\param pVkMemoryRequirements Memory requirements for each allocation.
\param pCreateInfo Creation parameters for each allocation.
\param allocationCount Number of allocations to make.
\param[out] pAllocations Pointer to array that will be filled with handles to created allocations.
\param[out] pAllocationInfo Optional. Pointer to array that will be filled with parameters of created allocations.

You should free the memory using vmaFreeMemory() or vmaFreeMemoryPages().

Word "pages" is just a suggestion to use this function to allocate pieces of memory needed for sparse binding.
It is just a general purpose allocation function able to make multiple allocations at once.
It may be internally optimized to be more efficient than calling vmaAllocateMemory() `allocationCount` times.

All allocations are made using same parameters. All of them are created out of the same memory pool and type.
If any allocation fails, all allocations already made within this function call are also freed, so that when
returned result is not `VK_SUCCESS`, `pAllocation` array is always entirely filled with `VK_NULL_HANDLE`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkMemoryRequirements* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pVkMemoryRequirements,
    const VmaAllocationCreateInfo* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pCreateInfo,
    size_t allocationCount,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations,
    VmaAllocationInfo* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) pAllocationInfo);

/** \brief Allocates memory suitable for given `VkBuffer`.

\param allocator
\param buffer
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

It only creates #VmaAllocation. To bind the memory to the buffer, use vmaBindBufferMemory().

This is a special-purpose function. In most cases you should use vmaCreateBuffer().

You must free the allocation using vmaFreeMemory() when no longer needed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Allocates memory suitable for given `VkImage`.

\param allocator
\param image
\param pCreateInfo
\param[out] pAllocation Handle to allocated memory.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

It only creates #VmaAllocation. To bind the memory to the buffer, use vmaBindImageMemory().

This is a special-purpose function. In most cases you should use vmaCreateImage().

You must free the allocation using vmaFreeMemory() when no longer needed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Frees memory previously allocated using vmaAllocateMemory(), vmaAllocateMemoryForBuffer(), or vmaAllocateMemoryForImage().

Passing `VK_NULL_HANDLE` as `allocation` is valid. Such function call is just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    const VmaAllocation VMA_NULLABLE allocation);

/** \brief Frees memory and destroys multiple allocations.

Word "pages" is just a suggestion to use this function to free pieces of memory used for sparse binding.
It is just a general purpose function to free memory and destroy allocations made using e.g. vmaAllocateMemory(),
vmaAllocateMemoryPages() and other functions.
It may be internally optimized to be more efficient than calling vmaFreeMemory() `allocationCount` times.

Allocations in `pAllocations` array can come from any memory pools and types.
Passing `VK_NULL_HANDLE` as elements of `pAllocations` array is valid. Such entries are just skipped.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator VMA_NOT_NULL allocator,
    size_t allocationCount,
    const VmaAllocation VMA_NULLABLE* VMA_NOT_NULL VMA_LEN_IF_NOT_NULL(allocationCount) pAllocations);

/** \brief Returns current information about specified allocation.

Current parameters of given allocation are returned in `pAllocationInfo`.

Although this function doesn't lock any mutex, so it should be quite efficient,
you should avoid calling it too often.
You can retrieve same VmaAllocationInfo structure while creating your resource, from function
vmaCreateBuffer(), vmaCreateImage(). You can remember it if you are sure parameters don't change
(e.g. due to defragmentation).
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VmaAllocationInfo* VMA_NOT_NULL pAllocationInfo);

/** \brief Sets pUserData in given allocation to new value.

The value of pointer `pUserData` is copied to allocation's `pUserData`.
It is opaque, so you can use it however you want - e.g.
as a pointer, ordinal number or some handle to you own data.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    void* VMA_NULLABLE pUserData);

/** \brief Sets pName in given allocation to new value.

`pName` must be either null, or pointer to a null-terminated string. The function
makes local copy of the string and sets it as allocation's `pName`. String
passed as pName doesn't need to be valid for whole lifetime of the allocation -
you can free it after this call. String previously pointed by allocation's
`pName` is freed from memory.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const char* VMA_NULLABLE pName);

/**
\brief Given an allocation, returns Property Flags of its memory type.

This is just a convenience function. Same information can be obtained using
vmaGetAllocationInfo() + vmaGetMemoryProperties().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags);

/** \brief Maps memory represented by given allocation and returns pointer to it.

Maps memory represented by given allocation to make it accessible to CPU code.
When succeeded, `*ppData` contains pointer to first byte of this memory.

\warning
If the allocation is part of a bigger `VkDeviceMemory` block, returned pointer is
correctly offsetted to the beginning of region assigned to this particular allocation.
Unlike the result of `vkMapMemory`, it points to the allocation, not to the beginning of the whole block.
You should not add VmaAllocationInfo::offset to it!

Mapping is internally reference-counted and synchronized, so despite raw Vulkan
function `vkMapMemory()` cannot be used to map same block of `VkDeviceMemory`
multiple times simultaneously, it is safe to call this function on allocations
assigned to the same memory block. Actual Vulkan memory will be mapped on first
mapping and unmapped on last unmapping.

If the function succeeded, you must call vmaUnmapMemory() to unmap the
allocation when mapping is no longer needed or before freeing the allocation, at
the latest.

It also safe to call this function multiple times on the same allocation. You
must call vmaUnmapMemory() same number of times as you called vmaMapMemory().

It is also safe to call this function on allocation created with
#VMA_ALLOCATION_CREATE_MAPPED_BIT flag. Its memory stays mapped all the time.
You must still call vmaUnmapMemory() same number of times as you called
vmaMapMemory(). You must not call vmaUnmapMemory() additional time to free the
"0-th" mapping made automatically due to #VMA_ALLOCATION_CREATE_MAPPED_BIT flag.

This function fails when used on allocation made in memory type that is not
`HOST_VISIBLE`.

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    void* VMA_NULLABLE* VMA_NOT_NULL ppData);

/** \brief Unmaps memory represented by given allocation, mapped previously using vmaMapMemory().

For details, see description of vmaMapMemory().

This function doesn't automatically flush or invalidate caches.
If the allocation is made from a memory types that is not `HOST_COHERENT`,
you also need to use vmaInvalidateAllocation() / vmaFlushAllocation(), as required by Vulkan specification.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation);

/** \brief Flushes memory of given allocation.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called after writing to a mapped memory for memory types that are not `HOST_COHERENT`.
Unmap operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Invalidates memory of given allocation.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given range of given allocation.
It needs to be called before reading from a mapped memory for memory types that are not `HOST_COHERENT`.
Map operation doesn't do that automatically.

- `offset` must be relative to the beginning of allocation.
- `size` can be `VK_WHOLE_SIZE`. It means all memory from `offset` the the end of given allocation.
- `offset` and `size` don't have to be aligned.
  They are internally rounded down/up to multiply of `nonCoherentAtomSize`.
- If `size` is 0, this call is ignored.
- If memory type that the `allocation` belongs to is not `HOST_VISIBLE` or it is `HOST_COHERENT`,
  this call is ignored.

Warning! `offset` and `size` are relative to the contents of given `allocation`.
If you mean whole allocation, you can pass 0 and `VK_WHOLE_SIZE`, respectively.
Do not pass allocation's offset as `offset`!!!

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if
it is called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize offset,
    VkDeviceSize size);

/** \brief Flushes memory of given set of allocations.

Calls `vkFlushMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of vmaFlushAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all ofsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkFlushMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocations(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const VmaAllocation VMA_NOT_NULL* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Invalidates memory of given set of allocations.

Calls `vkInvalidateMappedMemoryRanges()` for memory associated with given ranges of given allocations.
For more information, see documentation of vmaInvalidateAllocation().

\param allocator
\param allocationCount
\param allocations
\param offsets If not null, it must point to an array of offsets of regions to flush, relative to the beginning of respective allocations. Null means all ofsets are zero.
\param sizes If not null, it must point to an array of sizes of regions to flush in respective allocations. Null means `VK_WHOLE_SIZE` for all allocations.

This function returns the `VkResult` from `vkInvalidateMappedMemoryRanges` if it is
called, otherwise `VK_SUCCESS`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocations(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t allocationCount,
    const VmaAllocation VMA_NOT_NULL* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) allocations,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) offsets,
    const VkDeviceSize* VMA_NULLABLE VMA_LEN_IF_NOT_NULL(allocationCount) sizes);

/** \brief Checks magic number in margins around all allocations in given memory types (in both default and custom pools) in search for corruptions.

\param allocator
\param memoryTypeBits Bit mask, where each bit set means that a memory type with that index should be checked.

Corruption detection is enabled only when `VMA_DEBUG_DETECT_CORRUPTION` macro is defined to nonzero,
`VMA_DEBUG_MARGIN` is defined to nonzero and only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`. For more information, see [Corruption detection](@ref debugging_memory_usage_corruption_detection).

Possible return values:

- `VK_ERROR_FEATURE_NOT_PRESENT` - corruption detection is not enabled for any of specified memory types.
- `VK_SUCCESS` - corruption detection has been performed and succeeded.
- `VK_ERROR_UNKNOWN` - corruption detection has been performed and found memory corruptions around one of the allocations.
  `VMA_ASSERT` is also fired in that case.
- Other value: Error returned by Vulkan, e.g. memory mapping failure.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(
    VmaAllocator VMA_NOT_NULL allocator,
    uint32_t memoryTypeBits);

/** \brief Begins defragmentation process.

\param allocator Allocator object.
\param pInfo Structure filled with parameters of defragmentation.
\param[out] pContext Context object that must be passed to vmaEndDefragmentation() to finish defragmentation.
\returns
- `VK_SUCCESS` if defragmentation can begin.
- `VK_ERROR_FEATURE_NOT_PRESENT` if defragmentation is not supported.

For more information about defragmentation, see documentation chapter:
[Defragmentation](@ref defragmentation).
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentation(
    VmaAllocator VMA_NOT_NULL allocator,
    const VmaDefragmentationInfo* VMA_NOT_NULL pInfo,
    VmaDefragmentationContext VMA_NULLABLE* VMA_NOT_NULL pContext);

/** \brief Ends defragmentation process.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param[out] pStats Optional stats for the defragmentation. Can be null.

Use this function to finish defragmentation started by vmaBeginDefragmentation().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaEndDefragmentation(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationStats* VMA_NULLABLE pStats);

/** \brief Starts single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param[out] pPassInfo Computed information for current pass.
\returns
- `VK_SUCCESS` if no more moves are possible. Then you can omit call to vmaEndDefragmentationPass() and simply end whole defragmentation.
- `VK_INCOMPLETE` if there are pending moves returned in `pPassInfo`. You need to perform them, call vmaEndDefragmentationPass(),
  and then preferably try another pass with vmaBeginDefragmentationPass().
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo);

/** \brief Ends single defragmentation pass.

\param allocator Allocator object.
\param context Context object that has been created by vmaBeginDefragmentation().
\param pPassInfo Computed information for current pass filled by vmaBeginDefragmentationPass() and possibly modified by you.

Returns `VK_SUCCESS` if no more moves are possible or `VK_INCOMPLETE` if more defragmentations are possible.

Ends incremental defragmentation pass and commits all defragmentation moves from `pPassInfo`.
After this call:

- Allocations at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY
  (which is the default) will be pointing to the new destination place.
- Allocation at `pPassInfo[i].srcAllocation` that had `pPassInfo[i].operation ==` #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY
  will be freed.

If no more moves are possible you can end whole defragmentation.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaEndDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo);

/** \brief Binds buffer to allocation.

Binds specified buffer to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create a buffer, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindBufferMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateBuffer() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer);

/** \brief Binds buffer to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param buffer
\param pNext A chain of structures to be attached to `VkBindBufferMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindBufferMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer VMA_NOT_NULL_NON_DISPATCHABLE buffer,
    const void* VMA_NULLABLE pNext);

/** \brief Binds image to allocation.

Binds specified image to region of memory represented by specified allocation.
Gets `VkDeviceMemory` handle and offset from the allocation.
If you want to create an image, allocate memory for it and bind them together separately,
you should use this function for binding instead of standard `vkBindImageMemory()`,
because it ensures proper synchronization so that when a `VkDeviceMemory` object is used by multiple
allocations, calls to `vkBind*Memory()` or `vkMapMemory()` won't happen from multiple threads simultaneously
(which is illegal in Vulkan).

It is recommended to use function vmaCreateImage() instead of this one.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image);

/** \brief Binds image to allocation with additional parameters.

\param allocator
\param allocation
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the `allocation`. Normally it should be 0.
\param image
\param pNext A chain of structures to be attached to `VkBindImageMemoryInfoKHR` structure used internally. Normally it should be null.

This function is similar to vmaBindImageMemory(), but it provides additional parameters.

If `pNext` is not null, #VmaAllocator object must have been created with #VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT flag
or with VmaAllocatorCreateInfo::vulkanApiVersion `>= VK_API_VERSION_1_1`. Otherwise the call fails.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage VMA_NOT_NULL_NON_DISPATCHABLE image,
    const void* VMA_NULLABLE pNext);

/** \brief Creates a new `VkBuffer`, allocates and binds memory for it.

\param allocator
\param pBufferCreateInfo
\param pAllocationCreateInfo
\param[out] pBuffer Buffer that was created.
\param[out] pAllocation Allocation that was created.
\param[out] pAllocationInfo Optional. Information about allocated memory. It can be later fetched using function vmaGetAllocationInfo().

This function automatically:

-# Creates buffer.
-# Allocates appropriate memory for it.
-# Binds the buffer with the memory.

If any of these operations fail, buffer and allocation are not created,
returned value is negative error code, `*pBuffer` and `*pAllocation` are null.

If the function succeeded, you must destroy both buffer and allocation when you
no longer need them using either convenience function vmaDestroyBuffer() or
separately, using `vkDestroyBuffer()` and vmaFreeMemory().

If #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag was used,
VK_KHR_dedicated_allocation extension is used internally to query driver whether
it requires or prefers the new buffer to have dedicated allocation. If yes,
and if dedicated allocation is possible
(#VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT is not used), it creates dedicated
allocation for this buffer, just like when using
#VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.

\note This function creates a new `VkBuffer`. Sub-allocation of parts of one large buffer,
although recommended as a good practice, is out of scope of this library and could be implemented
by the user as a higher-level logic on top of VMA.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Creates a buffer with additional minimum alignment.

Similar to vmaCreateBuffer() but provides additional parameter `minAlignment` which allows to specify custom,
minimum alignment to be used when placing the buffer inside a larger memory block, which may be needed e.g.
for interop with OpenGL.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBufferWithAlignment(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param pBufferCreateInfo
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function vmaDestroyBuffer().

\note There is a new version of this function augmented with parameter `allocationLocalOffset` - see vmaCreateAliasingBuffer2().
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer);

/** \brief Creates a new `VkBuffer`, binds already created memory for it.

\param allocator
\param allocation Allocation that provides memory to be used for binding new buffer to it.
\param allocationLocalOffset Additional offset to be added while binding, relative to the beginning of the allocation. Normally it should be 0.
\param pBufferCreateInfo 
\param[out] pBuffer Buffer that was created.

This function automatically:

-# Creates buffer.
-# Binds the buffer with the supplied memory.

If any of these operations fail, buffer is not created,
returned value is negative error code and `*pBuffer` is null.

If the function succeeded, you must destroy the buffer when you
no longer need it using `vkDestroyBuffer()`. If you want to also destroy the corresponding
allocation you can use convenience function vmaDestroyBuffer().

\note This is a new version of the function augmented with parameter `allocationLocalOffset`.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer);

/** \brief Destroys Vulkan buffer and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyBuffer(device, buffer, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as buffer and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE buffer,
    VmaAllocation VMA_NULLABLE allocation);

/// Function similar to vmaCreateBuffer().
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator VMA_NOT_NULL allocator,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    const VmaAllocationCreateInfo* VMA_NOT_NULL pAllocationCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage,
    VmaAllocation VMA_NULLABLE* VMA_NOT_NULL pAllocation,
    VmaAllocationInfo* VMA_NULLABLE pAllocationInfo);

/// Function similar to vmaCreateAliasingBuffer() but for images.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage);

/// Function similar to vmaCreateAliasingBuffer2() but for images.
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage);

/** \brief Destroys Vulkan image and frees allocated memory.

This is just a convenience function equivalent to:

\code
vkDestroyImage(device, image, allocationCallbacks);
vmaFreeMemory(allocator, allocation);
\endcode

It is safe to pass null as image and/or allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE image,
    VmaAllocation VMA_NULLABLE allocation);

/** @} */

/**
\addtogroup group_virtual
@{
*/

/** \brief Creates new #VmaVirtualBlock object.

\param pCreateInfo Parameters for creation.
\param[out] pVirtualBlock Returned virtual block object or `VMA_NULL` if creation failed.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateVirtualBlock(
    const VmaVirtualBlockCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualBlock VMA_NULLABLE* VMA_NOT_NULL pVirtualBlock);

/** \brief Destroys #VmaVirtualBlock object.

Please note that you should consciously handle virtual allocations that could remain unfreed in the block.
You should either free them individually using vmaVirtualFree() or call vmaClearVirtualBlock()
if you are sure this is what you want. If you do neither, an assert is called.

If you keep pointers to some additional metadata associated with your virtual allocations in their `pUserData`,
don't forget to free them.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaDestroyVirtualBlock(
    VmaVirtualBlock VMA_NULLABLE virtualBlock);

/** \brief Returns true of the #VmaVirtualBlock is empty - contains 0 virtual allocations and has all its space available for new allocations.
*/
VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaIsVirtualBlockEmpty(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock);

/** \brief Returns information about a specific virtual allocation within a virtual block, like its size and `pUserData` pointer.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualAllocationInfo(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, VmaVirtualAllocationInfo* VMA_NOT_NULL pVirtualAllocInfo);

/** \brief Allocates new virtual allocation inside given #VmaVirtualBlock.

If the allocation fails due to not enough free space available, `VK_ERROR_OUT_OF_DEVICE_MEMORY` is returned
(despite the function doesn't ever allocate actual GPU memory).
`pAllocation` is then set to `VK_NULL_HANDLE` and `pOffset`, if not null, it set to `UINT64_MAX`.

\param virtualBlock Virtual block
\param pCreateInfo Parameters for the allocation
\param[out] pAllocation Returned handle of the new allocation
\param[out] pOffset Returned offset of the new allocation. Optional, can be null.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaVirtualAllocate(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    const VmaVirtualAllocationCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pAllocation,
    VkDeviceSize* VMA_NULLABLE pOffset);

/** \brief Frees virtual allocation inside given #VmaVirtualBlock.

It is correct to call this function with `allocation == VK_NULL_HANDLE` - it does nothing.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaVirtualFree(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE allocation);

/** \brief Frees all virtual allocations inside given #VmaVirtualBlock.

You must either call this function or free each virtual allocation individually with vmaVirtualFree()
before destroying a virtual block. Otherwise, an assert is called.

If you keep pointer to some additional metadata associated with your virtual allocation in its `pUserData`,
don't forget to free it as well.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaClearVirtualBlock(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock);

/** \brief Changes custom pointer associated with given virtual allocation.
*/
VMA_CALL_PRE void VMA_CALL_POST vmaSetVirtualAllocationUserData(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation,
    void* VMA_NULLABLE pUserData);

/** \brief Calculates and returns statistics about virtual allocations and memory usage in given #VmaVirtualBlock.

This function is fast to call. For more detailed statistics, see vmaCalculateVirtualBlockStatistics().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualBlockStatistics(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaStatistics* VMA_NOT_NULL pStats);

/** \brief Calculates and returns detailed statistics about virtual allocations and memory usage in given #VmaVirtualBlock.

This function is slow to call. Use for debugging purposes.
For less detailed statistics, see vmaGetVirtualBlockStatistics().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaCalculateVirtualBlockStatistics(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaDetailedStatistics* VMA_NOT_NULL pStats);

/** @} */

#if VMA_STATS_STRING_ENABLED
/**
\addtogroup group_stats
@{
*/

/** \brief Builds and returns a null-terminated string in JSON format with information about given #VmaVirtualBlock.
\param virtualBlock Virtual block.
\param[out] ppStatsString Returned string.
\param detailedMap Pass `VK_FALSE` to only obtain statistics as returned by vmaCalculateVirtualBlockStatistics(). Pass `VK_TRUE` to also obtain full list of allocations and free spaces.

Returned string must be freed using vmaFreeVirtualBlockStatsString().
*/
VMA_CALL_PRE void VMA_CALL_POST vmaBuildVirtualBlockStatsString(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE* VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

/// Frees a string returned by vmaBuildVirtualBlockStatsString().
VMA_CALL_PRE void VMA_CALL_POST vmaFreeVirtualBlockStatsString(
    VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE pStatsString);

/** \brief Builds and returns statistics as a null-terminated string in JSON format.
\param allocator
\param[out] ppStatsString Must be freed using vmaFreeStatsString() function.
\param detailedMap
*/
VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator VMA_NOT_NULL allocator,
    char* VMA_NULLABLE* VMA_NOT_NULL ppStatsString,
    VkBool32 detailedMap);

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator VMA_NOT_NULL allocator,
    char* VMA_NULLABLE pStatsString);

/** @} */

#endif // VMA_STATS_STRING_ENABLED

#endif // _VMA_FUNCTION_HEADERS

#ifdef __cplusplus
}
#endif

#endif // AMD_VULKAN_MEMORY_ALLOCATOR_H

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//    IMPLEMENTATION
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// For Visual Studio IntelliSense.
#if defined(__cplusplus) && defined(__INTELLISENSE__)
#define VMA_IMPLEMENTATION
#endif

#ifdef VMA_IMPLEMENTATION
#undef VMA_IMPLEMENTATION

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <type_traits>

#ifdef _MSC_VER
    #include <intrin.h> // For functions like __popcnt, _BitScanForward etc.
#endif
#if __cplusplus >= 202002L || _MSVC_LANG >= 202002L // C++20
    #include <bit> // For std::popcount
#endif

#if VMA_STATS_STRING_ENABLED
    #include <cstdio> // For snprintf
#endif

/*******************************************************************************
CONFIGURATION SECTION

Define some of these macros before each #include of this header or change them
here if you need other then default behavior depending on your environment.
*/
#ifndef _VMA_CONFIGURATION

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = &vkAllocateMemory;
*/
#if !defined(VMA_STATIC_VULKAN_FUNCTIONS) && !defined(VK_NO_PROTOTYPES)
    #define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif

/*
Define this macro to 1 to make the library fetch pointers to Vulkan functions
internally, like:

    vulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");

To use this feature in new versions of VMA you now have to pass
VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as
VmaAllocatorCreateInfo::pVulkanFunctions. Other members can be null.
*/
#if !defined(VMA_DYNAMIC_VULKAN_FUNCTIONS)
    #define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#ifndef VMA_USE_STL_SHARED_MUTEX
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define VMA_USE_STL_SHARED_MUTEX 1
    // Visual studio defines __cplusplus properly only when passed additional parameter: /Zc:__cplusplus
    // Otherwise it is always 199711L, despite shared_mutex works since Visual Studio 2015 Update 2.
    #elif defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 190023918 && __cplusplus == 199711L && _MSVC_LANG >= 201703L
        #define VMA_USE_STL_SHARED_MUTEX 1
    #else
        #define VMA_USE_STL_SHARED_MUTEX 0
    #endif
#endif

/*
Define this macro to include custom header files without having to edit this file directly, e.g.:

    // Inside of "my_vma_configuration_user_includes.h":

    #include "my_custom_assert.h" // for MY_CUSTOM_ASSERT
    #include "my_custom_min.h" // for my_custom_min
    #include <algorithm>
    #include <mutex>

    // Inside a different file, which includes "vk_mem_alloc.h":

    #define VMA_CONFIGURATION_USER_INCLUDES_H "my_vma_configuration_user_includes.h"
    #define VMA_ASSERT(expr) MY_CUSTOM_ASSERT(expr)
    #define VMA_MIN(v1, v2)  (my_custom_min(v1, v2))
    #include "vk_mem_alloc.h"
    ...

The following headers are used in this CONFIGURATION section only, so feel free to
remove them if not needed.
*/
#if !defined(VMA_CONFIGURATION_USER_INCLUDES_H)
    #include <cassert> // for assert
    #include <algorithm> // for min, max
    #include <mutex>
#else
    #include VMA_CONFIGURATION_USER_INCLUDES_H
#endif

#ifndef VMA_NULL
   // Value used as null pointer. Define it to e.g.: nullptr, NULL, 0, (void*)0.
   #define VMA_NULL   nullptr
#endif

#ifndef VMA_FALLTHROUGH
    #if __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
        #define VMA_FALLTHROUGH [[fallthrough]]
    #else
        #define VMA_FALLTHROUGH
    #endif
#endif

// Normal assert to check for programmer's errors, especially in Debug configuration.
#ifndef VMA_ASSERT
   #ifdef NDEBUG
       #define VMA_ASSERT(expr)
   #else
       #define VMA_ASSERT(expr)         assert(expr)
   #endif
#endif

// Assert that will be called very often, like inside data structures e.g. operator[].
// Making it non-empty can make program slow.
#ifndef VMA_HEAVY_ASSERT
   #ifdef NDEBUG
       #define VMA_HEAVY_ASSERT(expr)
   #else
       #define VMA_HEAVY_ASSERT(expr)   //VMA_ASSERT(expr)
   #endif
#endif

// If your compiler is not compatible with C++17 and definition of
// aligned_alloc() function is missing, uncommenting following line may help:

//#include <malloc.h>

#if defined(__ANDROID_API__) && (__ANDROID_API__ < 16)
#include <cstdlib>
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    return memalign(alignment, size);
}
#elif defined(__APPLE__) || defined(__ANDROID__) || (defined(__linux__) && defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#include <cstdlib>

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    // Unfortunately, aligned_alloc causes VMA to crash due to it returning null pointers. (At least under 11.4)
    // Therefore, for now disable this specific exception until a proper solution is found.
    //#if defined(__APPLE__) && (defined(MAC_OS_X_VERSION_10_16) || defined(__IPHONE_14_0))
    //#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_16 || __IPHONE_OS_VERSION_MAX_ALLOWED >= __IPHONE_14_0
    //    // For C++14, usr/include/malloc/_malloc.h declares aligned_alloc()) only
    //    // with the MacOSX11.0 SDK in Xcode 12 (which is what adds
    //    // MAC_OS_X_VERSION_10_16), even though the function is marked
    //    // available for 10.15. That is why the preprocessor checks for 10.16 but
    //    // the __builtin_available checks for 10.15.
    //    // People who use C++17 could call aligned_alloc with the 10.15 SDK already.
    //    if (__builtin_available(macOS 10.15, iOS 13, *))
    //        return aligned_alloc(alignment, size);
    //#endif
    //#endif

    // alignment must be >= sizeof(void*)
    if(alignment < sizeof(void*))
    {
        alignment = sizeof(void*);
    }

    void *pointer;
    if(posix_memalign(&pointer, alignment, size) == 0)
        return pointer;
    return VMA_NULL;
}
#elif defined(_WIN32)
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    return _aligned_malloc(size, alignment);
}
#elif __cplusplus >= 201703L || _MSVC_LANG >= 201703L // C++17
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    return aligned_alloc(alignment, size);
}
#else
static void* vma_aligned_alloc(size_t alignment, size_t size)
{
    VMA_ASSERT(0 && "Could not implement aligned_alloc automatically. Please enable C++17 or later in your compiler or provide custom implementation of macro VMA_SYSTEM_ALIGNED_MALLOC (and VMA_SYSTEM_ALIGNED_FREE if needed) using the API of your system.");
    return VMA_NULL;
}
#endif

#if defined(_WIN32)
static void vma_aligned_free(void* ptr)
{
    _aligned_free(ptr);
}
#else
static void vma_aligned_free(void* VMA_NULLABLE ptr)
{
    free(ptr);
}
#endif

#ifndef VMA_ALIGN_OF
   #define VMA_ALIGN_OF(type)       (alignof(type))
#endif

#ifndef VMA_SYSTEM_ALIGNED_MALLOC
   #define VMA_SYSTEM_ALIGNED_MALLOC(size, alignment) vma_aligned_alloc((alignment), (size))
#endif

#ifndef VMA_SYSTEM_ALIGNED_FREE
   // VMA_SYSTEM_FREE is the old name, but might have been defined by the user
   #if defined(VMA_SYSTEM_FREE)
      #define VMA_SYSTEM_ALIGNED_FREE(ptr)     VMA_SYSTEM_FREE(ptr)
   #else
      #define VMA_SYSTEM_ALIGNED_FREE(ptr)     vma_aligned_free(ptr)
    #endif
#endif

#ifndef VMA_COUNT_BITS_SET
    // Returns number of bits set to 1 in (v)
    #define VMA_COUNT_BITS_SET(v) VmaCountBitsSet(v)
#endif

#ifndef VMA_BITSCAN_LSB
    // Scans integer for index of first nonzero value from the Least Significant Bit (LSB). If mask is 0 then returns UINT8_MAX
    #define VMA_BITSCAN_LSB(mask) VmaBitScanLSB(mask)
#endif

#ifndef VMA_BITSCAN_MSB
    // Scans integer for index of first nonzero value from the Most Significant Bit (MSB). If mask is 0 then returns UINT8_MAX
    #define VMA_BITSCAN_MSB(mask) VmaBitScanMSB(mask)
#endif

#ifndef VMA_MIN
   #define VMA_MIN(v1, v2)    ((std::min)((v1), (v2)))
#endif

#ifndef VMA_MAX
   #define VMA_MAX(v1, v2)    ((std::max)((v1), (v2)))
#endif

#ifndef VMA_SWAP
   #define VMA_SWAP(v1, v2)   std::swap((v1), (v2))
#endif

#ifndef VMA_SORT
   #define VMA_SORT(beg, end, cmp)  std::sort(beg, end, cmp)
#endif

#ifndef VMA_DEBUG_LOG_FORMAT
   #define VMA_DEBUG_LOG_FORMAT(format, ...)
   /*
   #define VMA_DEBUG_LOG_FORMAT(format, ...) do { \
       printf((format), __VA_ARGS__); \
       printf("\n"); \
   } while(false)
   */
#endif

#ifndef VMA_DEBUG_LOG
    #define VMA_DEBUG_LOG(str)   VMA_DEBUG_LOG_FORMAT("%s", (str))
#endif

#ifndef VMA_CLASS_NO_COPY
    #define VMA_CLASS_NO_COPY(className) \
        private: \
            className(const className&) = delete; \
            className& operator=(const className&) = delete;
#endif
#ifndef VMA_CLASS_NO_COPY_NO_MOVE
    #define VMA_CLASS_NO_COPY_NO_MOVE(className) \
        private: \
            className(const className&) = delete; \
            className(className&&) = delete; \
            className& operator=(const className&) = delete; \
            className& operator=(className&&) = delete;
#endif

// Define this macro to 1 to enable functions: vmaBuildStatsString, vmaFreeStatsString.
#if VMA_STATS_STRING_ENABLED
    static inline void VmaUint32ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint32_t num)
    {
        snprintf(outStr, strLen, "%u", static_cast<unsigned int>(num));
    }
    static inline void VmaUint64ToStr(char* VMA_NOT_NULL outStr, size_t strLen, uint64_t num)
    {
        snprintf(outStr, strLen, "%llu", static_cast<unsigned long long>(num));
    }
    static inline void VmaPtrToStr(char* VMA_NOT_NULL outStr, size_t strLen, const void* ptr)
    {
        snprintf(outStr, strLen, "%p", ptr);
    }
#endif

#ifndef VMA_MUTEX
    class VmaMutex
    {
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutex)
    public:
        VmaMutex() { }
        void Lock() { m_Mutex.lock(); }
        void Unlock() { m_Mutex.unlock(); }
        bool TryLock() { return m_Mutex.try_lock(); }
    private:
        std::mutex m_Mutex;
    };
    #define VMA_MUTEX VmaMutex
#endif

// Read-write mutex, where "read" is shared access, "write" is exclusive access.
#ifndef VMA_RW_MUTEX
    #if VMA_USE_STL_SHARED_MUTEX
        // Use std::shared_mutex from C++17.
        #include <shared_mutex>
        class VmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.lock_shared(); }
            void UnlockRead() { m_Mutex.unlock_shared(); }
            bool TryLockRead() { return m_Mutex.try_lock_shared(); }
            void LockWrite() { m_Mutex.lock(); }
            void UnlockWrite() { m_Mutex.unlock(); }
            bool TryLockWrite() { return m_Mutex.try_lock(); }
        private:
            std::shared_mutex m_Mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #elif defined(_WIN32) && defined(WINVER) && WINVER >= 0x0600
        // Use SRWLOCK from WinAPI.
        // Minimum supported client = Windows Vista, server = Windows Server 2008.
        class VmaRWMutex
        {
        public:
            VmaRWMutex() { InitializeSRWLock(&m_Lock); }
            void LockRead() { AcquireSRWLockShared(&m_Lock); }
            void UnlockRead() { ReleaseSRWLockShared(&m_Lock); }
            bool TryLockRead() { return TryAcquireSRWLockShared(&m_Lock) != FALSE; }
            void LockWrite() { AcquireSRWLockExclusive(&m_Lock); }
            void UnlockWrite() { ReleaseSRWLockExclusive(&m_Lock); }
            bool TryLockWrite() { return TryAcquireSRWLockExclusive(&m_Lock) != FALSE; }
        private:
            SRWLOCK m_Lock;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #else
        // Less efficient fallback: Use normal mutex.
        class VmaRWMutex
        {
        public:
            void LockRead() { m_Mutex.Lock(); }
            void UnlockRead() { m_Mutex.Unlock(); }
            bool TryLockRead() { return m_Mutex.TryLock(); }
            void LockWrite() { m_Mutex.Lock(); }
            void UnlockWrite() { m_Mutex.Unlock(); }
            bool TryLockWrite() { return m_Mutex.TryLock(); }
        private:
            VMA_MUTEX m_Mutex;
        };
        #define VMA_RW_MUTEX VmaRWMutex
    #endif // #if VMA_USE_STL_SHARED_MUTEX
#endif // #ifndef VMA_RW_MUTEX

/*
If providing your own implementation, you need to implement a subset of std::atomic.
*/
#ifndef VMA_ATOMIC_UINT32
    #include <atomic>
    #define VMA_ATOMIC_UINT32 std::atomic<uint32_t>
#endif

#ifndef VMA_ATOMIC_UINT64
    #include <atomic>
    #define VMA_ATOMIC_UINT64 std::atomic<uint64_t>
#endif

#ifndef VMA_DEBUG_ALWAYS_DEDICATED_MEMORY
    /**
    Every allocation will have its own memory block.
    Define to 1 for debugging purposes only.
    */
    #define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY (0)
#endif

#ifndef VMA_MIN_ALIGNMENT
    /**
    Minimum alignment of all allocations, in bytes.
    Set to more than 1 for debugging purposes. Must be power of two.
    */
    #ifdef VMA_DEBUG_ALIGNMENT // Old name
        #define VMA_MIN_ALIGNMENT VMA_DEBUG_ALIGNMENT
    #else
        #define VMA_MIN_ALIGNMENT (1)
    #endif
#endif

#ifndef VMA_DEBUG_MARGIN
    /**
    Minimum margin after every allocation, in bytes.
    Set nonzero for debugging purposes only.
    */
    #define VMA_DEBUG_MARGIN (0)
#endif

#ifndef VMA_DEBUG_INITIALIZE_ALLOCATIONS
    /**
    Define this macro to 1 to automatically fill new allocations and destroyed
    allocations with some bit pattern.
    */
    #define VMA_DEBUG_INITIALIZE_ALLOCATIONS (0)
#endif

#ifndef VMA_DEBUG_DETECT_CORRUPTION
    /**
    Define this macro to 1 together with non-zero value of VMA_DEBUG_MARGIN to
    enable writing magic value to the margin after every allocation and
    validating it, so that memory corruptions (out-of-bounds writes) are detected.
    */
    #define VMA_DEBUG_DETECT_CORRUPTION (0)
#endif

#ifndef VMA_DEBUG_GLOBAL_MUTEX
    /**
    Set this to 1 for debugging purposes only, to enable single mutex protecting all
    entry calls to the library. Can be useful for debugging multithreading issues.
    */
    #define VMA_DEBUG_GLOBAL_MUTEX (0)
#endif

#ifndef VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY
    /**
    Minimum value for VkPhysicalDeviceLimits::bufferImageGranularity.
    Set to more than 1 for debugging purposes only. Must be power of two.
    */
    #define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY (1)
#endif

#ifndef VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
    /*
    Set this to 1 to make VMA never exceed VkPhysicalDeviceLimits::maxMemoryAllocationCount
    and return error instead of leaving up to Vulkan implementation what to do in such cases.
    */
    #define VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT (0)
#endif

#ifndef VMA_SMALL_HEAP_MAX_SIZE
   /// Maximum size of a memory heap in Vulkan to consider it "small".
   #define VMA_SMALL_HEAP_MAX_SIZE (1024ull * 1024 * 1024)
#endif

#ifndef VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE
   /// Default size of a block allocated as single VkDeviceMemory from a "large" heap.
   #define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (256ull * 1024 * 1024)
#endif

/*
Mapping hysteresis is a logic that launches when vmaMapMemory/vmaUnmapMemory is called
or a persistently mapped allocation is created and destroyed several times in a row.
It keeps additional +1 mapping of a device memory block to prevent calling actual
vkMapMemory/vkUnmapMemory too many times, which may improve performance and help
tools like RenderDoc.
*/
#ifndef VMA_MAPPING_HYSTERESIS_ENABLED
    #define VMA_MAPPING_HYSTERESIS_ENABLED 1
#endif

#define VMA_VALIDATE(cond) do { if(!(cond)) { \
        VMA_ASSERT(0 && "Validation failed: " #cond); \
        return false; \
    } } while(false)

/*******************************************************************************
END OF CONFIGURATION
*/
#endif // _VMA_CONFIGURATION


static const uint8_t VMA_ALLOCATION_FILL_PATTERN_CREATED = 0xDC;
static const uint8_t VMA_ALLOCATION_FILL_PATTERN_DESTROYED = 0xEF;
// Decimal 2139416166, float NaN, little-endian binary 66 E6 84 7F.
static const uint32_t VMA_CORRUPTION_DETECTION_MAGIC_VALUE = 0x7F84E666;

// Copy of some Vulkan definitions so we don't need to check their existence just to handle few constants.
static const uint32_t VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY = 0x00000040;
static const uint32_t VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY = 0x00000080;
static const uint32_t VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY = 0x00020000;
static const uint32_t VK_IMAGE_CREATE_DISJOINT_BIT_COPY = 0x00000200;
static const int32_t VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY = 1000158000;
static const uint32_t VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET = 0x10000000u;
static const uint32_t VMA_ALLOCATION_TRY_COUNT = 32;
static const uint32_t VMA_VENDOR_ID_AMD = 4098;

// This one is tricky. Vulkan specification defines this code as available since
// Vulkan 1.0, but doesn't actually define it in Vulkan SDK earlier than 1.2.131.
// See pull request #207.
#define VK_ERROR_UNKNOWN_COPY ((VkResult)-13)


#if VMA_STATS_STRING_ENABLED
// Correspond to values of enum VmaSuballocationType.
static const char* VMA_SUBALLOCATION_TYPE_NAMES[] =
{
    "FREE",
    "UNKNOWN",
    "BUFFER",
    "IMAGE_UNKNOWN",
    "IMAGE_LINEAR",
    "IMAGE_OPTIMAL",
};
#endif

static VkAllocationCallbacks VmaEmptyAllocationCallbacks =
    { VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL, VMA_NULL };


#ifndef _VMA_ENUM_DECLARATIONS

enum VmaSuballocationType
{
    VMA_SUBALLOCATION_TYPE_FREE = 0,
    VMA_SUBALLOCATION_TYPE_UNKNOWN = 1,
    VMA_SUBALLOCATION_TYPE_BUFFER = 2,
    VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN = 3,
    VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR = 4,
    VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL = 5,
    VMA_SUBALLOCATION_TYPE_MAX_ENUM = 0x7FFFFFFF
};

enum VMA_CACHE_OPERATION
{
    VMA_CACHE_FLUSH,
    VMA_CACHE_INVALIDATE
};

enum class VmaAllocationRequestType
{
    Normal,
    TLSF,
    // Used by "Linear" algorithm.
    UpperAddress,
    EndOf1st,
    EndOf2nd,
};

#endif // _VMA_ENUM_DECLARATIONS

#ifndef _VMA_FORWARD_DECLARATIONS
// Opaque handle used by allocation algorithms to identify single allocation in any conforming way.
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VmaAllocHandle)

struct VmaMutexLock;
struct VmaMutexLockRead;
struct VmaMutexLockWrite;

template<typename T>
struct AtomicTransactionalIncrement;

template<typename T>
struct VmaStlAllocator;

template<typename T, typename AllocatorT>
class VmaVector;

template<typename T, typename AllocatorT, size_t N>
class VmaSmallVector;

template<typename T>
class VmaPoolAllocator;

template<typename T>
struct VmaListItem;

template<typename T>
class VmaRawList;

template<typename T, typename AllocatorT>
class VmaList;

template<typename ItemTypeTraits>
class VmaIntrusiveLinkedList;

// Unused in this version
#if 0
template<typename T1, typename T2>
struct VmaPair;
template<typename FirstT, typename SecondT>
struct VmaPairFirstLess;

template<typename KeyT, typename ValueT>
class VmaMap;
#endif

#if VMA_STATS_STRING_ENABLED
class VmaStringBuilder;
class VmaJsonWriter;
#endif

class VmaDeviceMemoryBlock;

struct VmaDedicatedAllocationListItemTraits;
class VmaDedicatedAllocationList;

struct VmaSuballocation;
struct VmaSuballocationOffsetLess;
struct VmaSuballocationOffsetGreater;
struct VmaSuballocationItemSizeLess;

typedef VmaList<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> VmaSuballocationList;

struct VmaAllocationRequest;

class VmaBlockMetadata;
class VmaBlockMetadata_Linear;
class VmaBlockMetadata_TLSF;

class VmaBlockVector;

struct VmaPoolListItemTraits;

struct VmaCurrentBudgetData;

class VmaAllocationObjectAllocator;

#endif // _VMA_FORWARD_DECLARATIONS


#ifndef _VMA_FUNCTIONS

/*
Returns number of bits set to 1 in (v).

On specific platforms and compilers you can use instrinsics like:

Visual Studio:
    return __popcnt(v);
GCC, Clang:
    return static_cast<uint32_t>(__builtin_popcount(v));

Define macro VMA_COUNT_BITS_SET to provide your optimized implementation.
But you need to check in runtime whether user's CPU supports these, as some old processors don't.
*/
static inline uint32_t VmaCountBitsSet(uint32_t v)
{
#if __cplusplus >= 202002L || _MSVC_LANG >= 202002L // C++20
    return std::popcount(v);
#else
    uint32_t c = v - ((v >> 1) & 0x55555555);
    c = ((c >> 2) & 0x33333333) + (c & 0x33333333);
    c = ((c >> 4) + c) & 0x0F0F0F0F;
    c = ((c >> 8) + c) & 0x00FF00FF;
    c = ((c >> 16) + c) & 0x0000FFFF;
    return c;
#endif
}

static inline uint8_t VmaBitScanLSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanForward64(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffsll(mask)) - 1U;
#else
    uint8_t pos = 0;
    uint64_t bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 63);
    return UINT8_MAX;
#endif
}

static inline uint8_t VmaBitScanLSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanForward(&pos, mask))
        return static_cast<uint8_t>(pos);
    return UINT8_MAX;
#elif defined __GNUC__ || defined __clang__
    return static_cast<uint8_t>(__builtin_ffs(mask)) - 1U;
#else
    uint8_t pos = 0;
    uint32_t bit = 1;
    do
    {
        if (mask & bit)
            return pos;
        bit <<= 1;
    } while (pos++ < 31);
    return UINT8_MAX;
#endif
}

static inline uint8_t VmaBitScanMSB(uint64_t mask)
{
#if defined(_MSC_VER) && defined(_WIN64)
    unsigned long pos;
    if (_BitScanReverse64(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif defined __GNUC__ || defined __clang__
    if (mask)
        return 63 - static_cast<uint8_t>(__builtin_clzll(mask));
#else
    uint8_t pos = 63;
    uint64_t bit = 1ULL << 63;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

static inline uint8_t VmaBitScanMSB(uint32_t mask)
{
#ifdef _MSC_VER
    unsigned long pos;
    if (_BitScanReverse(&pos, mask))
        return static_cast<uint8_t>(pos);
#elif defined __GNUC__ || defined __clang__
    if (mask)
        return 31 - static_cast<uint8_t>(__builtin_clz(mask));
#else
    uint8_t pos = 31;
    uint32_t bit = 1UL << 31;
    do
    {
        if (mask & bit)
            return pos;
        bit >>= 1;
    } while (pos-- > 0);
#endif
    return UINT8_MAX;
}

/*
Returns true if given number is a power of two.
T must be unsigned integer number or signed integer but always nonnegative.
For 0 returns true.
*/
template <typename T>
inline bool VmaIsPow2(T x)
{
    return (x & (x - 1)) == 0;
}

// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignUp(T val, T alignment)
{
    VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
    return (val + alignment - 1) & ~(alignment - 1);
}

// Aligns given value down to nearest multiply of align value. For example: VmaAlignDown(11, 8) = 8.
// Use types like uint32_t, uint64_t as T.
template <typename T>
static inline T VmaAlignDown(T val, T alignment)
{
    VMA_HEAVY_ASSERT(VmaIsPow2(alignment));
    return val & ~(alignment - 1);
}

// Division with mathematical rounding to nearest number.
template <typename T>
static inline T VmaRoundDiv(T x, T y)
{
    return (x + (y / (T)2)) / y;
}

// Divide by 'y' and round up to nearest integer.
template <typename T>
static inline T VmaDivideRoundingUp(T x, T y)
{
    return (x + y - (T)1) / y;
}

// Returns smallest power of 2 greater or equal to v.
static inline uint32_t VmaNextPow2(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static inline uint64_t VmaNextPow2(uint64_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

// Returns largest power of 2 less or equal to v.
static inline uint32_t VmaPrevPow2(uint32_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v = v ^ (v >> 1);
    return v;
}

static inline uint64_t VmaPrevPow2(uint64_t v)
{
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v = v ^ (v >> 1);
    return v;
}

static inline bool VmaStrIsEmpty(const char* pStr)
{
    return pStr == VMA_NULL || *pStr == '\0';
}

/*
Returns true if two memory blocks occupy overlapping pages.
ResourceA must be in less memory offset than ResourceB.

Algorithm is based on "Vulkan 1.0.39 - A Specification (with all registered Vulkan extensions)"
chapter 11.6 "Resource Memory Association", paragraph "Buffer-Image Granularity".
*/
static inline bool VmaBlocksOnSamePage(
    VkDeviceSize resourceAOffset,
    VkDeviceSize resourceASize,
    VkDeviceSize resourceBOffset,
    VkDeviceSize pageSize)
{
    VMA_ASSERT(resourceAOffset + resourceASize <= resourceBOffset && resourceASize > 0 && pageSize > 0);
    VkDeviceSize resourceAEnd = resourceAOffset + resourceASize - 1;
    VkDeviceSize resourceAEndPage = resourceAEnd & ~(pageSize - 1);
    VkDeviceSize resourceBStart = resourceBOffset;
    VkDeviceSize resourceBStartPage = resourceBStart & ~(pageSize - 1);
    return resourceAEndPage == resourceBStartPage;
}

/*
Returns true if given suballocation types could conflict and must respect
VkPhysicalDeviceLimits::bufferImageGranularity. They conflict if one is buffer
or linear image and another one is optimal image. If type is unknown, behave
conservatively.
*/
static inline bool VmaIsBufferImageGranularityConflict(
    VmaSuballocationType suballocType1,
    VmaSuballocationType suballocType2)
{
    if (suballocType1 > suballocType2)
    {
        VMA_SWAP(suballocType1, suballocType2);
    }

    switch (suballocType1)
    {
    case VMA_SUBALLOCATION_TYPE_FREE:
        return false;
    case VMA_SUBALLOCATION_TYPE_UNKNOWN:
        return true;
    case VMA_SUBALLOCATION_TYPE_BUFFER:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR ||
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR:
        return
            suballocType2 == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL;
    case VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL:
        return false;
    default:
        VMA_ASSERT(0);
        return true;
    }
}

static void VmaWriteMagicValue(void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    uint32_t* pDst = (uint32_t*)((char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pDst)
    {
        *pDst = VMA_CORRUPTION_DETECTION_MAGIC_VALUE;
    }
#else
    // no-op
#endif
}

static bool VmaValidateMagicValue(const void* pData, VkDeviceSize offset)
{
#if VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_DETECT_CORRUPTION
    const uint32_t* pSrc = (const uint32_t*)((const char*)pData + offset);
    const size_t numberCount = VMA_DEBUG_MARGIN / sizeof(uint32_t);
    for (size_t i = 0; i < numberCount; ++i, ++pSrc)
    {
        if (*pSrc != VMA_CORRUPTION_DETECTION_MAGIC_VALUE)
        {
            return false;
        }
    }
#endif
    return true;
}

/*
Fills structure with parameters of an example buffer to be used for transfers
during GPU memory defragmentation.
*/
static void VmaFillGpuDefragmentationBufferCreateInfo(VkBufferCreateInfo& outBufCreateInfo)
{
    memset(&outBufCreateInfo, 0, sizeof(outBufCreateInfo));
    outBufCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    outBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    outBufCreateInfo.size = (VkDeviceSize)VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE; // Example size.
}


/*
Performs binary search and returns iterator to first element that is greater or
equal to (key), according to comparison (cmp).

Cmp should return true if first argument is less than second argument.

Returned value is the found element, if present in the collection or place where
new element with value (key) should be inserted.
*/
template <typename CmpLess, typename IterT, typename KeyT>
static IterT VmaBinaryFindFirstNotLess(IterT beg, IterT end, const KeyT& key, const CmpLess& cmp)
{
    size_t down = 0, up = size_t(end - beg);
    while (down < up)
    {
        const size_t mid = down + (up - down) / 2;  // Overflow-safe midpoint calculation
        if (cmp(*(beg + mid), key))
        {
            down = mid + 1;
        }
        else
        {
            up = mid;
        }
    }
    return beg + down;
}

template<typename CmpLess, typename IterT, typename KeyT>
IterT VmaBinaryFindSorted(const IterT& beg, const IterT& end, const KeyT& value, const CmpLess& cmp)
{
    IterT it = VmaBinaryFindFirstNotLess<CmpLess, IterT, KeyT>(
        beg, end, value, cmp);
    if (it == end ||
        (!cmp(*it, value) && !cmp(value, *it)))
    {
        return it;
    }
    return end;
}

/*
Returns true if all pointers in the array are not-null and unique.
Warning! O(n^2) complexity. Use only inside VMA_HEAVY_ASSERT.
T must be pointer type, e.g. VmaAllocation, VmaPool.
*/
template<typename T>
static bool VmaValidatePointerArray(uint32_t count, const T* arr)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        const T iPtr = arr[i];
        if (iPtr == VMA_NULL)
        {
            return false;
        }
        for (uint32_t j = i + 1; j < count; ++j)
        {
            if (iPtr == arr[j])
            {
                return false;
            }
        }
    }
    return true;
}

template<typename MainT, typename NewT>
static inline void VmaPnextChainPushFront(MainT* mainStruct, NewT* newStruct)
{
    newStruct->pNext = mainStruct->pNext;
    mainStruct->pNext = newStruct;
}

// This is the main algorithm that guides the selection of a memory type best for an allocation -
// converts usage to required/preferred/not preferred flags.
static bool FindMemoryPreferences(
    bool isIntegratedGPU,
    const VmaAllocationCreateInfo& allocCreateInfo,
    VkFlags bufImgUsage, // VkBufferCreateInfo::usage or VkImageCreateInfo::usage. UINT32_MAX if unknown.
    VkMemoryPropertyFlags& outRequiredFlags,
    VkMemoryPropertyFlags& outPreferredFlags,
    VkMemoryPropertyFlags& outNotPreferredFlags)
{
    outRequiredFlags = allocCreateInfo.requiredFlags;
    outPreferredFlags = allocCreateInfo.preferredFlags;
    outNotPreferredFlags = 0;

    switch(allocCreateInfo.usage)
    {
    case VMA_MEMORY_USAGE_UNKNOWN:
        break;
    case VMA_MEMORY_USAGE_GPU_ONLY:
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_CPU_ONLY:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_TO_GPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        if(!isIntegratedGPU || (outPreferredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
        {
            outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case VMA_MEMORY_USAGE_GPU_TO_CPU:
        outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        outPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case VMA_MEMORY_USAGE_CPU_COPY:
        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED:
        outRequiredFlags |= VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
        break;
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
    {
        if(bufImgUsage == UINT32_MAX)
        {
            VMA_ASSERT(0 && "VMA_MEMORY_USAGE_AUTO* values can only be used with functions like vmaCreateBuffer, vmaCreateImage so that the details of the created resource are known.");
            return false;
        }
        // This relies on values of VK_IMAGE_USAGE_TRANSFER* being the same VK_BUFFER_IMAGE_TRANSFER*.
        const bool deviceAccess = (bufImgUsage & ~(VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) != 0;
        const bool hostAccessSequentialWrite = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) != 0;
        const bool hostAccessRandom = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) != 0;
        const bool hostAccessAllowTransferInstead = (allocCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) != 0;
        const bool preferDevice = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        const bool preferHost = allocCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

        // CPU random access - e.g. a buffer written to or transferred from GPU to read back on CPU.
        if(hostAccessRandom)
        {
            if(!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
            {
                // Nice if it will end up in HOST_VISIBLE, but more importantly prefer DEVICE_LOCAL.
                // Omitting HOST_VISIBLE here is intentional.
                // In case there is DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED, it will pick that one.
                // Otherwise, this will give same weight to DEVICE_LOCAL as HOST_VISIBLE | HOST_CACHED and select the former if occurs first on the list.
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            }
            else
            {
                // Always CPU memory, cached.
                outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            }
        }
        // CPU sequential write - may be CPU or host-visible GPU memory, uncached and write-combined.
        else if(hostAccessSequentialWrite)
        {
            // Want uncached and write-combined.
            outNotPreferredFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

            if(!isIntegratedGPU && deviceAccess && hostAccessAllowTransferInstead && !preferHost)
            {
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            }
            else
            {
                outRequiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                // Direct GPU access, CPU sequential write (e.g. a dynamic uniform buffer updated every frame)
                if(deviceAccess)
                {
                    // Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose GPU memory.
                    if(preferHost)
                        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    else
                        outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                }
                // GPU no direct access, CPU sequential write (e.g. an upload buffer to be transferred to the GPU)
                else
                {
                    // Could go to CPU memory or GPU BAR/unified. Up to the user to decide. If no preference, choose CPU memory.
                    if(preferDevice)
                        outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    else
                        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                }
            }
        }
        // No CPU access
        else
        {
            // GPU access, no CPU access (e.g. a color attachment image) - prefer GPU memory
            if(deviceAccess)
            {
                // ...unless there is a clear preference from the user not to do so.
                if(preferHost)
                    outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                else
                    outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            }
            // No direct GPU access, no CPU access, just transfers.
            // It may be staging copy intended for e.g. preserving image for next frame (then better GPU memory) or
            // a "swap file" copy to free some GPU memory (then better CPU memory).
            // Up to the user to decide. If no preferece, assume the former and choose GPU memory.
            if(preferHost)
                outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            else
                outPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    }
    default:
        VMA_ASSERT(0);
    }

    // Avoid DEVICE_COHERENT unless explicitly requested.
    if(((allocCreateInfo.requiredFlags | allocCreateInfo.preferredFlags) &
        (VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)) == 0)
    {
        outNotPreferredFlags |= VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Memory allocation

static void* VmaMalloc(const VkAllocationCallbacks* pAllocationCallbacks, size_t size, size_t alignment)
{
    void* result = VMA_NULL;
    if ((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnAllocation != VMA_NULL))
    {
        result = (*pAllocationCallbacks->pfnAllocation)(
            pAllocationCallbacks->pUserData,
            size,
            alignment,
            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
    }
    else
    {
        result = VMA_SYSTEM_ALIGNED_MALLOC(size, alignment);
    }
    VMA_ASSERT(result != VMA_NULL && "CPU memory allocation failed.");
    return result;
}

static void VmaFree(const VkAllocationCallbacks* pAllocationCallbacks, void* ptr)
{
    if ((pAllocationCallbacks != VMA_NULL) &&
        (pAllocationCallbacks->pfnFree != VMA_NULL))
    {
        (*pAllocationCallbacks->pfnFree)(pAllocationCallbacks->pUserData, ptr);
    }
    else
    {
        VMA_SYSTEM_ALIGNED_FREE(ptr);
    }
}

template<typename T>
static T* VmaAllocate(const VkAllocationCallbacks* pAllocationCallbacks)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VmaAllocateArray(const VkAllocationCallbacks* pAllocationCallbacks, size_t count)
{
    return (T*)VmaMalloc(pAllocationCallbacks, sizeof(T) * count, VMA_ALIGN_OF(T));
}

#define vma_new(allocator, type)   new(VmaAllocate<type>(allocator))(type)

#define vma_new_array(allocator, type, count)   new(VmaAllocateArray<type>((allocator), (count)))(type)

template<typename T>
static void vma_delete(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr)
{
    ptr->~T();
    VmaFree(pAllocationCallbacks, ptr);
}

template<typename T>
static void vma_delete_array(const VkAllocationCallbacks* pAllocationCallbacks, T* ptr, size_t count)
{
    if (ptr != VMA_NULL)
    {
        for (size_t i = count; i--; )
        {
            ptr[i].~T();
        }
        VmaFree(pAllocationCallbacks, ptr);
    }
}

static char* VmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr)
{
    if (srcStr != VMA_NULL)
    {
        const size_t len = strlen(srcStr);
        char* const result = vma_new_array(allocs, char, len + 1);
        memcpy(result, srcStr, len + 1);
        return result;
    }
    return VMA_NULL;
}

#if VMA_STATS_STRING_ENABLED
static char* VmaCreateStringCopy(const VkAllocationCallbacks* allocs, const char* srcStr, size_t strLen)
{
    if (srcStr != VMA_NULL)
    {
        char* const result = vma_new_array(allocs, char, strLen + 1);
        memcpy(result, srcStr, strLen);
        result[strLen] = '\0';
        return result;
    }
    return VMA_NULL;
}
#endif // VMA_STATS_STRING_ENABLED

static void VmaFreeString(const VkAllocationCallbacks* allocs, char* str)
{
    if (str != VMA_NULL)
    {
        const size_t len = strlen(str);
        vma_delete_array(allocs, str, len + 1);
    }
}

template<typename CmpLess, typename VectorT>
size_t VmaVectorInsertSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    const size_t indexToInsert = VmaBinaryFindFirstNotLess(
        vector.data(),
        vector.data() + vector.size(),
        value,
        CmpLess()) - vector.data();
    VmaVectorInsert(vector, indexToInsert, value);
    return indexToInsert;
}

template<typename CmpLess, typename VectorT>
bool VmaVectorRemoveSorted(VectorT& vector, const typename VectorT::value_type& value)
{
    CmpLess comparator;
    typename VectorT::iterator it = VmaBinaryFindFirstNotLess(
        vector.begin(),
        vector.end(),
        value,
        comparator);
    if ((it != vector.end()) && !comparator(*it, value) && !comparator(value, *it))
    {
        size_t indexToRemove = it - vector.begin();
        VmaVectorRemove(vector, indexToRemove);
        return true;
    }
    return false;
}
#endif // _VMA_FUNCTIONS

#ifndef _VMA_STATISTICS_FUNCTIONS

static void VmaClearStatistics(VmaStatistics& outStats)
{
    outStats.blockCount = 0;
    outStats.allocationCount = 0;
    outStats.blockBytes = 0;
    outStats.allocationBytes = 0;
}

static void VmaAddStatistics(VmaStatistics& inoutStats, const VmaStatistics& src)
{
    inoutStats.blockCount += src.blockCount;
    inoutStats.allocationCount += src.allocationCount;
    inoutStats.blockBytes += src.blockBytes;
    inoutStats.allocationBytes += src.allocationBytes;
}

static void VmaClearDetailedStatistics(VmaDetailedStatistics& outStats)
{
    VmaClearStatistics(outStats.statistics);
    outStats.unusedRangeCount = 0;
    outStats.allocationSizeMin = VK_WHOLE_SIZE;
    outStats.allocationSizeMax = 0;
    outStats.unusedRangeSizeMin = VK_WHOLE_SIZE;
    outStats.unusedRangeSizeMax = 0;
}

static void VmaAddDetailedStatisticsAllocation(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.statistics.allocationCount++;
    inoutStats.statistics.allocationBytes += size;
    inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, size);
    inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, size);
}

static void VmaAddDetailedStatisticsUnusedRange(VmaDetailedStatistics& inoutStats, VkDeviceSize size)
{
    inoutStats.unusedRangeCount++;
    inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, size);
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, size);
}

static void VmaAddDetailedStatistics(VmaDetailedStatistics& inoutStats, const VmaDetailedStatistics& src)
{
    VmaAddStatistics(inoutStats.statistics, src.statistics);
    inoutStats.unusedRangeCount += src.unusedRangeCount;
    inoutStats.allocationSizeMin = VMA_MIN(inoutStats.allocationSizeMin, src.allocationSizeMin);
    inoutStats.allocationSizeMax = VMA_MAX(inoutStats.allocationSizeMax, src.allocationSizeMax);
    inoutStats.unusedRangeSizeMin = VMA_MIN(inoutStats.unusedRangeSizeMin, src.unusedRangeSizeMin);
    inoutStats.unusedRangeSizeMax = VMA_MAX(inoutStats.unusedRangeSizeMax, src.unusedRangeSizeMax);
}

#endif // _VMA_STATISTICS_FUNCTIONS

#ifndef _VMA_MUTEX_LOCK
// Helper RAII class to lock a mutex in constructor and unlock it in destructor (at the end of scope).
struct VmaMutexLock
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLock)
public:
    VmaMutexLock(VMA_MUTEX& mutex, bool useMutex = true) :
        m_pMutex(useMutex ? &mutex : VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->Lock(); }
    }
    ~VmaMutexLock() {  if (m_pMutex) { m_pMutex->Unlock(); } }

private:
    VMA_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for reading.
struct VmaMutexLockRead
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockRead)
public:
    VmaMutexLockRead(VMA_RW_MUTEX& mutex, bool useMutex) :
        m_pMutex(useMutex ? &mutex : VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->LockRead(); }
    }
    ~VmaMutexLockRead() { if (m_pMutex) { m_pMutex->UnlockRead(); } }

private:
    VMA_RW_MUTEX* m_pMutex;
};

// Helper RAII class to lock a RW mutex in constructor and unlock it in destructor (at the end of scope), for writing.
struct VmaMutexLockWrite
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMutexLockWrite)
public:
    VmaMutexLockWrite(VMA_RW_MUTEX& mutex, bool useMutex)
        : m_pMutex(useMutex ? &mutex : VMA_NULL)
    {
        if (m_pMutex) { m_pMutex->LockWrite(); }
    }
    ~VmaMutexLockWrite() { if (m_pMutex) { m_pMutex->UnlockWrite(); } }

private:
    VMA_RW_MUTEX* m_pMutex;
};

#if VMA_DEBUG_GLOBAL_MUTEX
    static VMA_MUTEX gDebugGlobalMutex;
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK VmaMutexLock debugGlobalMutexLock(gDebugGlobalMutex, true);
#else
    #define VMA_DEBUG_GLOBAL_MUTEX_LOCK
#endif
#endif // _VMA_MUTEX_LOCK

#ifndef _VMA_ATOMIC_TRANSACTIONAL_INCREMENT
// An object that increments given atomic but decrements it back in the destructor unless Commit() is called.
template<typename AtomicT>
struct AtomicTransactionalIncrement
{
public:
    using T = decltype(AtomicT().load());

    ~AtomicTransactionalIncrement()
    {
        if(m_Atomic)
            --(*m_Atomic);
    }

    void Commit() { m_Atomic = nullptr; }
    T Increment(AtomicT* atomic)
    {
        m_Atomic = atomic;
        return m_Atomic->fetch_add(1);
    }

private:
    AtomicT* m_Atomic = nullptr;
};
#endif // _VMA_ATOMIC_TRANSACTIONAL_INCREMENT

#ifndef _VMA_STL_ALLOCATOR
// STL-compatible allocator.
template<typename T>
struct VmaStlAllocator
{
    const VkAllocationCallbacks* const m_pCallbacks;
    typedef T value_type;

    VmaStlAllocator(const VkAllocationCallbacks* pCallbacks) : m_pCallbacks(pCallbacks) {}
    template<typename U>
    VmaStlAllocator(const VmaStlAllocator<U>& src) : m_pCallbacks(src.m_pCallbacks) {}
    VmaStlAllocator(const VmaStlAllocator&) = default;
    VmaStlAllocator& operator=(const VmaStlAllocator&) = delete;

    T* allocate(size_t n) { return VmaAllocateArray<T>(m_pCallbacks, n); }
    void deallocate(T* p, size_t n) { VmaFree(m_pCallbacks, p); }

    template<typename U>
    bool operator==(const VmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks == rhs.m_pCallbacks;
    }
    template<typename U>
    bool operator!=(const VmaStlAllocator<U>& rhs) const
    {
        return m_pCallbacks != rhs.m_pCallbacks;
    }
};
#endif // _VMA_STL_ALLOCATOR

#ifndef _VMA_VECTOR
/* Class with interface compatible with subset of std::vector.
T must be POD because constructors and destructors are not called and memcpy is
used for these objects. */
template<typename T, typename AllocatorT>
class VmaVector
{
public:
    typedef T value_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    VmaVector(const AllocatorT& allocator);
    VmaVector(size_t count, const AllocatorT& allocator);
    // This version of the constructor is here for compatibility with pre-C++14 std::vector.
    // value is unused.
    VmaVector(size_t count, const T& value, const AllocatorT& allocator) : VmaVector(count, allocator) {}
    VmaVector(const VmaVector<T, AllocatorT>& src);
    VmaVector& operator=(const VmaVector& rhs);
    ~VmaVector() { VmaFree(m_Allocator.m_pCallbacks, m_pArray); }

    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_pArray; }
    T& front() { VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[0]; }
    T& back() { VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[m_Count - 1]; }
    const T* data() const { return m_pArray; }
    const T& front() const { VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[0]; }
    const T& back() const { VMA_HEAVY_ASSERT(m_Count > 0); return m_pArray[m_Count - 1]; }

    iterator begin() { return m_pArray; }
    iterator end() { return m_pArray + m_Count; }
    const_iterator cbegin() const { return m_pArray; }
    const_iterator cend() const { return m_pArray + m_Count; }
    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    void pop_front() { VMA_HEAVY_ASSERT(m_Count > 0); remove(0); }
    void pop_back() { VMA_HEAVY_ASSERT(m_Count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void reserve(size_t newCapacity, bool freeMemory = false);
    void resize(size_t newCount);
    void clear() { resize(0); }
    void shrink_to_fit();
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { VMA_HEAVY_ASSERT(index < m_Count); return m_pArray[index]; }
    const T& operator[](size_t index) const { VMA_HEAVY_ASSERT(index < m_Count); return m_pArray[index]; }

private:
    AllocatorT m_Allocator;
    T* m_pArray;
    size_t m_Count;
    size_t m_Capacity;
};

#ifndef _VMA_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(const AllocatorT& allocator)
    : m_Allocator(allocator),
    m_pArray(VMA_NULL),
    m_Count(0),
    m_Capacity(0) {}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(size_t count, const AllocatorT& allocator)
    : m_Allocator(allocator),
    m_pArray(count ? (T*)VmaAllocateArray<T>(allocator.m_pCallbacks, count) : VMA_NULL),
    m_Count(count),
    m_Capacity(count) {}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>::VmaVector(const VmaVector& src)
    : m_Allocator(src.m_Allocator),
    m_pArray(src.m_Count ? (T*)VmaAllocateArray<T>(src.m_Allocator.m_pCallbacks, src.m_Count) : VMA_NULL),
    m_Count(src.m_Count),
    m_Capacity(src.m_Count)
{
    if (m_Count != 0)
    {
        memcpy(m_pArray, src.m_pArray, m_Count * sizeof(T));
    }
}

template<typename T, typename AllocatorT>
VmaVector<T, AllocatorT>& VmaVector<T, AllocatorT>::operator=(const VmaVector& rhs)
{
    if (&rhs != this)
    {
        resize(rhs.m_Count);
        if (m_Count != 0)
        {
            memcpy(m_pArray, rhs.m_pArray, m_Count * sizeof(T));
        }
    }
    return *this;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    m_pArray[newIndex] = src;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::reserve(size_t newCapacity, bool freeMemory)
{
    newCapacity = VMA_MAX(newCapacity, m_Count);

    if ((newCapacity < m_Capacity) && !freeMemory)
    {
        newCapacity = m_Capacity;
    }

    if (newCapacity != m_Capacity)
    {
        T* const newArray = newCapacity ? VmaAllocateArray<T>(m_Allocator, newCapacity) : VMA_NULL;
        if (m_Count != 0)
        {
            memcpy(newArray, m_pArray, m_Count * sizeof(T));
        }
        VmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = newCapacity;
        m_pArray = newArray;
    }
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::resize(size_t newCount)
{
    size_t newCapacity = m_Capacity;
    if (newCount > m_Capacity)
    {
        newCapacity = VMA_MAX(newCount, VMA_MAX(m_Capacity * 3 / 2, (size_t)8));
    }

    if (newCapacity != m_Capacity)
    {
        T* const newArray = newCapacity ? VmaAllocateArray<T>(m_Allocator.m_pCallbacks, newCapacity) : VMA_NULL;
        const size_t elementsToCopy = VMA_MIN(m_Count, newCount);
        if (elementsToCopy != 0)
        {
            memcpy(newArray, m_pArray, elementsToCopy * sizeof(T));
        }
        VmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = newCapacity;
        m_pArray = newArray;
    }

    m_Count = newCount;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::shrink_to_fit()
{
    if (m_Capacity > m_Count)
    {
        T* newArray = VMA_NULL;
        if (m_Count > 0)
        {
            newArray = VmaAllocateArray<T>(m_Allocator.m_pCallbacks, m_Count);
            memcpy(newArray, m_pArray, m_Count * sizeof(T));
        }
        VmaFree(m_Allocator.m_pCallbacks, m_pArray);
        m_Capacity = m_Count;
        m_pArray = newArray;
    }
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::insert(size_t index, const T& src)
{
    VMA_HEAVY_ASSERT(index <= m_Count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    if (index < oldCount)
    {
        memmove(m_pArray + (index + 1), m_pArray + index, (oldCount - index) * sizeof(T));
    }
    m_pArray[index] = src;
}

template<typename T, typename AllocatorT>
void VmaVector<T, AllocatorT>::remove(size_t index)
{
    VMA_HEAVY_ASSERT(index < m_Count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        memmove(m_pArray + index, m_pArray + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _VMA_VECTOR_FUNCTIONS

template<typename T, typename allocatorT>
static void VmaVectorInsert(VmaVector<T, allocatorT>& vec, size_t index, const T& item)
{
    vec.insert(index, item);
}

template<typename T, typename allocatorT>
static void VmaVectorRemove(VmaVector<T, allocatorT>& vec, size_t index)
{
    vec.remove(index);
}
#endif // _VMA_VECTOR

#ifndef _VMA_SMALL_VECTOR
/*
This is a vector (a variable-sized array), optimized for the case when the array is small.

It contains some number of elements in-place, which allows it to avoid heap allocation
when the actual number of elements is below that threshold. This allows normal "small"
cases to be fast without losing generality for large inputs.
*/
template<typename T, typename AllocatorT, size_t N>
class VmaSmallVector
{
public:
    typedef T value_type;
    typedef T* iterator;

    VmaSmallVector(const AllocatorT& allocator);
    VmaSmallVector(size_t count, const AllocatorT& allocator);
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    VmaSmallVector(const VmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    template<typename SrcT, typename SrcAllocatorT, size_t SrcN>
    VmaSmallVector<T, AllocatorT, N>& operator=(const VmaSmallVector<SrcT, SrcAllocatorT, SrcN>&) = delete;
    ~VmaSmallVector() = default;

    bool empty() const { return m_Count == 0; }
    size_t size() const { return m_Count; }
    T* data() { return m_Count > N ? m_DynamicArray.data() : m_StaticArray; }
    T& front() { VMA_HEAVY_ASSERT(m_Count > 0); return data()[0]; }
    T& back() { VMA_HEAVY_ASSERT(m_Count > 0); return data()[m_Count - 1]; }
    const T* data() const { return m_Count > N ? m_DynamicArray.data() : m_StaticArray; }
    const T& front() const { VMA_HEAVY_ASSERT(m_Count > 0); return data()[0]; }
    const T& back() const { VMA_HEAVY_ASSERT(m_Count > 0); return data()[m_Count - 1]; }

    iterator begin() { return data(); }
    iterator end() { return data() + m_Count; }

    void pop_front() { VMA_HEAVY_ASSERT(m_Count > 0); remove(0); }
    void pop_back() { VMA_HEAVY_ASSERT(m_Count > 0); resize(size() - 1); }
    void push_front(const T& src) { insert(0, src); }

    void push_back(const T& src);
    void resize(size_t newCount, bool freeMemory = false);
    void clear(bool freeMemory = false);
    void insert(size_t index, const T& src);
    void remove(size_t index);

    T& operator[](size_t index) { VMA_HEAVY_ASSERT(index < m_Count); return data()[index]; }
    const T& operator[](size_t index) const { VMA_HEAVY_ASSERT(index < m_Count); return data()[index]; }

private:
    size_t m_Count;
    T m_StaticArray[N]; // Used when m_Size <= N
    VmaVector<T, AllocatorT> m_DynamicArray; // Used when m_Size > N
};

#ifndef _VMA_SMALL_VECTOR_FUNCTIONS
template<typename T, typename AllocatorT, size_t N>
VmaSmallVector<T, AllocatorT, N>::VmaSmallVector(const AllocatorT& allocator)
    : m_Count(0),
    m_DynamicArray(allocator) {}

template<typename T, typename AllocatorT, size_t N>
VmaSmallVector<T, AllocatorT, N>::VmaSmallVector(size_t count, const AllocatorT& allocator)
    : m_Count(count),
    m_DynamicArray(count > N ? count : 0, allocator) {}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::push_back(const T& src)
{
    const size_t newIndex = size();
    resize(newIndex + 1);
    data()[newIndex] = src;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::resize(size_t newCount, bool freeMemory)
{
    if (newCount > N && m_Count > N)
    {
        // Any direction, staying in m_DynamicArray
        m_DynamicArray.resize(newCount);
        if (freeMemory)
        {
            m_DynamicArray.shrink_to_fit();
        }
    }
    else if (newCount > N && m_Count <= N)
    {
        // Growing, moving from m_StaticArray to m_DynamicArray
        m_DynamicArray.resize(newCount);
        if (m_Count > 0)
        {
            memcpy(m_DynamicArray.data(), m_StaticArray, m_Count * sizeof(T));
        }
    }
    else if (newCount <= N && m_Count > N)
    {
        // Shrinking, moving from m_DynamicArray to m_StaticArray
        if (newCount > 0)
        {
            memcpy(m_StaticArray, m_DynamicArray.data(), newCount * sizeof(T));
        }
        m_DynamicArray.resize(0);
        if (freeMemory)
        {
            m_DynamicArray.shrink_to_fit();
        }
    }
    else
    {
        // Any direction, staying in m_StaticArray - nothing to do here
    }
    m_Count = newCount;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::clear(bool freeMemory)
{
    m_DynamicArray.clear();
    if (freeMemory)
    {
        m_DynamicArray.shrink_to_fit();
    }
    m_Count = 0;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::insert(size_t index, const T& src)
{
    VMA_HEAVY_ASSERT(index <= m_Count);
    const size_t oldCount = size();
    resize(oldCount + 1);
    T* const dataPtr = data();
    if (index < oldCount)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from m_StaticArray to m_DynamicArray.
        memmove(dataPtr + (index + 1), dataPtr + index, (oldCount - index) * sizeof(T));
    }
    dataPtr[index] = src;
}

template<typename T, typename AllocatorT, size_t N>
void VmaSmallVector<T, AllocatorT, N>::remove(size_t index)
{
    VMA_HEAVY_ASSERT(index < m_Count);
    const size_t oldCount = size();
    if (index < oldCount - 1)
    {
        //  I know, this could be more optimal for case where memmove can be memcpy directly from m_DynamicArray to m_StaticArray.
        T* const dataPtr = data();
        memmove(dataPtr + index, dataPtr + (index + 1), (oldCount - index - 1) * sizeof(T));
    }
    resize(oldCount - 1);
}
#endif // _VMA_SMALL_VECTOR_FUNCTIONS
#endif // _VMA_SMALL_VECTOR

#ifndef _VMA_POOL_ALLOCATOR
/*
Allocator for objects of type T using a list of arrays (pools) to speed up
allocation. Number of elements that can be allocated is not bounded because
allocator can create multiple blocks.
*/
template<typename T>
class VmaPoolAllocator
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaPoolAllocator)
public:
    VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity);
    ~VmaPoolAllocator();
    template<typename... Types> T* Alloc(Types&&... args);
    void Free(T* ptr);

private:
    union Item
    {
        uint32_t NextFreeIndex;
        alignas(T) char Value[sizeof(T)];
    };
    struct ItemBlock
    {
        Item* pItems;
        uint32_t Capacity;
        uint32_t FirstFreeIndex;
    };

    const VkAllocationCallbacks* m_pAllocationCallbacks;
    const uint32_t m_FirstBlockCapacity;
    VmaVector<ItemBlock, VmaStlAllocator<ItemBlock>> m_ItemBlocks;

    ItemBlock& CreateNewBlock();
};

#ifndef _VMA_POOL_ALLOCATOR_FUNCTIONS
template<typename T>
VmaPoolAllocator<T>::VmaPoolAllocator(const VkAllocationCallbacks* pAllocationCallbacks, uint32_t firstBlockCapacity)
    : m_pAllocationCallbacks(pAllocationCallbacks),
    m_FirstBlockCapacity(firstBlockCapacity),
    m_ItemBlocks(VmaStlAllocator<ItemBlock>(pAllocationCallbacks))
{
    VMA_ASSERT(m_FirstBlockCapacity > 1);
}

template<typename T>
VmaPoolAllocator<T>::~VmaPoolAllocator()
{
    for (size_t i = m_ItemBlocks.size(); i--;)
        vma_delete_array(m_pAllocationCallbacks, m_ItemBlocks[i].pItems, m_ItemBlocks[i].Capacity);
    m_ItemBlocks.clear();
}

template<typename T>
template<typename... Types> T* VmaPoolAllocator<T>::Alloc(Types&&... args)
{
    for (size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];
        // This block has some free items: Use first one.
        if (block.FirstFreeIndex != UINT32_MAX)
        {
            Item* const pItem = &block.pItems[block.FirstFreeIndex];
            block.FirstFreeIndex = pItem->NextFreeIndex;
            T* result = (T*)&pItem->Value;
            new(result)T(std::forward<Types>(args)...); // Explicit constructor call.
            return result;
        }
    }

    // No block has free item: Create new one and use it.
    ItemBlock& newBlock = CreateNewBlock();
    Item* const pItem = &newBlock.pItems[0];
    newBlock.FirstFreeIndex = pItem->NextFreeIndex;
    T* result = (T*)&pItem->Value;
    new(result) T(std::forward<Types>(args)...); // Explicit constructor call.
    return result;
}

template<typename T>
void VmaPoolAllocator<T>::Free(T* ptr)
{
    // Search all memory blocks to find ptr.
    for (size_t i = m_ItemBlocks.size(); i--; )
    {
        ItemBlock& block = m_ItemBlocks[i];

        // Casting to union.
        Item* pItemPtr;
        memcpy(&pItemPtr, &ptr, sizeof(pItemPtr));

        // Check if pItemPtr is in address range of this block.
        if ((pItemPtr >= block.pItems) && (pItemPtr < block.pItems + block.Capacity))
        {
            ptr->~T(); // Explicit destructor call.
            const uint32_t index = static_cast<uint32_t>(pItemPtr - block.pItems);
            pItemPtr->NextFreeIndex = block.FirstFreeIndex;
            block.FirstFreeIndex = index;
            return;
        }
    }
    VMA_ASSERT(0 && "Pointer doesn't belong to this memory pool.");
}

template<typename T>
typename VmaPoolAllocator<T>::ItemBlock& VmaPoolAllocator<T>::CreateNewBlock()
{
    const uint32_t newBlockCapacity = m_ItemBlocks.empty() ?
        m_FirstBlockCapacity : m_ItemBlocks.back().Capacity * 3 / 2;

    const ItemBlock newBlock =
    {
        vma_new_array(m_pAllocationCallbacks, Item, newBlockCapacity),
        newBlockCapacity,
        0
    };

    m_ItemBlocks.push_back(newBlock);

    // Setup singly-linked list of all free items in this block.
    for (uint32_t i = 0; i < newBlockCapacity - 1; ++i)
        newBlock.pItems[i].NextFreeIndex = i + 1;
    newBlock.pItems[newBlockCapacity - 1].NextFreeIndex = UINT32_MAX;
    return m_ItemBlocks.back();
}
#endif // _VMA_POOL_ALLOCATOR_FUNCTIONS
#endif // _VMA_POOL_ALLOCATOR

#ifndef _VMA_RAW_LIST
template<typename T>
struct VmaListItem
{
    VmaListItem* pPrev;
    VmaListItem* pNext;
    T Value;
};

// Doubly linked list.
template<typename T>
class VmaRawList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaRawList)
public:
    typedef VmaListItem<T> ItemType;

    VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks);
    // Intentionally not calling Clear, because that would be unnecessary
    // computations to return all items to m_ItemAllocator as free.
    ~VmaRawList() = default;

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }

    ItemType* Front() { return m_pFront; }
    ItemType* Back() { return m_pBack; }
    const ItemType* Front() const { return m_pFront; }
    const ItemType* Back() const { return m_pBack; }

    ItemType* PushFront();
    ItemType* PushBack();
    ItemType* PushFront(const T& value);
    ItemType* PushBack(const T& value);
    void PopFront();
    void PopBack();

    // Item can be null - it means PushBack.
    ItemType* InsertBefore(ItemType* pItem);
    // Item can be null - it means PushFront.
    ItemType* InsertAfter(ItemType* pItem);
    ItemType* InsertBefore(ItemType* pItem, const T& value);
    ItemType* InsertAfter(ItemType* pItem, const T& value);

    void Clear();
    void Remove(ItemType* pItem);

private:
    const VkAllocationCallbacks* const m_pAllocationCallbacks;
    VmaPoolAllocator<ItemType> m_ItemAllocator;
    ItemType* m_pFront;
    ItemType* m_pBack;
    size_t m_Count;
};

#ifndef _VMA_RAW_LIST_FUNCTIONS
template<typename T>
VmaRawList<T>::VmaRawList(const VkAllocationCallbacks* pAllocationCallbacks)
    : m_pAllocationCallbacks(pAllocationCallbacks),
    m_ItemAllocator(pAllocationCallbacks, 128),
    m_pFront(VMA_NULL),
    m_pBack(VMA_NULL),
    m_Count(0) {}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushFront()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pPrev = VMA_NULL;
    if (IsEmpty())
    {
        pNewItem->pNext = VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pNext = m_pFront;
        m_pFront->pPrev = pNewItem;
        m_pFront = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushBack()
{
    ItemType* const pNewItem = m_ItemAllocator.Alloc();
    pNewItem->pNext = VMA_NULL;
    if(IsEmpty())
    {
        pNewItem->pPrev = VMA_NULL;
        m_pFront = pNewItem;
        m_pBack = pNewItem;
        m_Count = 1;
    }
    else
    {
        pNewItem->pPrev = m_pBack;
        m_pBack->pNext = pNewItem;
        m_pBack = pNewItem;
        ++m_Count;
    }
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushFront(const T& value)
{
    ItemType* const pNewItem = PushFront();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::PushBack(const T& value)
{
    ItemType* const pNewItem = PushBack();
    pNewItem->Value = value;
    return pNewItem;
}

template<typename T>
void VmaRawList<T>::PopFront()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pFrontItem = m_pFront;
    ItemType* const pNextItem = pFrontItem->pNext;
    if (pNextItem != VMA_NULL)
    {
        pNextItem->pPrev = VMA_NULL;
    }
    m_pFront = pNextItem;
    m_ItemAllocator.Free(pFrontItem);
    --m_Count;
}

template<typename T>
void VmaRawList<T>::PopBack()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const pBackItem = m_pBack;
    ItemType* const pPrevItem = pBackItem->pPrev;
    if(pPrevItem != VMA_NULL)
    {
        pPrevItem->pNext = VMA_NULL;
    }
    m_pBack = pPrevItem;
    m_ItemAllocator.Free(pBackItem);
    --m_Count;
}

template<typename T>
void VmaRawList<T>::Clear()
{
    if (IsEmpty() == false)
    {
        ItemType* pItem = m_pBack;
        while (pItem != VMA_NULL)
        {
            ItemType* const pPrevItem = pItem->pPrev;
            m_ItemAllocator.Free(pItem);
            pItem = pPrevItem;
        }
        m_pFront = VMA_NULL;
        m_pBack = VMA_NULL;
        m_Count = 0;
    }
}

template<typename T>
void VmaRawList<T>::Remove(ItemType* pItem)
{
    VMA_HEAVY_ASSERT(pItem != VMA_NULL);
    VMA_HEAVY_ASSERT(m_Count > 0);

    if(pItem->pPrev != VMA_NULL)
    {
        pItem->pPrev->pNext = pItem->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(m_pFront == pItem);
        m_pFront = pItem->pNext;
    }

    if(pItem->pNext != VMA_NULL)
    {
        pItem->pNext->pPrev = pItem->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(m_pBack == pItem);
        m_pBack = pItem->pPrev;
    }

    m_ItemAllocator.Free(pItem);
    --m_Count;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertBefore(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const prevItem = pItem->pPrev;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pPrev = prevItem;
        newItem->pNext = pItem;
        pItem->pPrev = newItem;
        if(prevItem != VMA_NULL)
        {
            prevItem->pNext = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_pFront == pItem);
            m_pFront = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
        return PushBack();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertAfter(ItemType* pItem)
{
    if(pItem != VMA_NULL)
    {
        ItemType* const nextItem = pItem->pNext;
        ItemType* const newItem = m_ItemAllocator.Alloc();
        newItem->pNext = nextItem;
        newItem->pPrev = pItem;
        pItem->pNext = newItem;
        if(nextItem != VMA_NULL)
        {
            nextItem->pPrev = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_pBack == pItem);
            m_pBack = newItem;
        }
        ++m_Count;
        return newItem;
    }
    else
        return PushFront();
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertBefore(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertBefore(pItem);
    newItem->Value = value;
    return newItem;
}

template<typename T>
VmaListItem<T>* VmaRawList<T>::InsertAfter(ItemType* pItem, const T& value)
{
    ItemType* const newItem = InsertAfter(pItem);
    newItem->Value = value;
    return newItem;
}
#endif // _VMA_RAW_LIST_FUNCTIONS
#endif // _VMA_RAW_LIST

#ifndef _VMA_LIST
template<typename T, typename AllocatorT>
class VmaList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaList)
public:
    class reverse_iterator;
    class const_iterator;
    class const_reverse_iterator;

    class iterator
    {
        friend class const_iterator;
        friend class VmaList<T, AllocatorT>;
    public:
        iterator() :  m_pList(VMA_NULL), m_pItem(VMA_NULL) {}
        iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        T& operator*() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return m_pItem->Value; }
        T* operator->() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return &m_pItem->Value; }

        bool operator==(const iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        iterator operator++(int) { iterator result = *this; ++*this; return result; }
        iterator operator--(int) { iterator result = *this; --*this; return result; }

        iterator& operator++() { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); m_pItem = m_pItem->pNext; return *this; }
        iterator& operator--();

    private:
        VmaRawList<T>* m_pList;
        VmaListItem<T>* m_pItem;

        iterator(VmaRawList<T>* pList, VmaListItem<T>* pItem) : m_pList(pList),  m_pItem(pItem) {}
    };
    class reverse_iterator
    {
        friend class const_reverse_iterator;
        friend class VmaList<T, AllocatorT>;
    public:
        reverse_iterator() : m_pList(VMA_NULL), m_pItem(VMA_NULL) {}
        reverse_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        T& operator*() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return m_pItem->Value; }
        T* operator->() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return &m_pItem->Value; }

        bool operator==(const reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        reverse_iterator operator++(int) { reverse_iterator result = *this; ++* this; return result; }
        reverse_iterator operator--(int) { reverse_iterator result = *this; --* this; return result; }

        reverse_iterator& operator++() { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); m_pItem = m_pItem->pPrev; return *this; }
        reverse_iterator& operator--();

    private:
        VmaRawList<T>* m_pList;
        VmaListItem<T>* m_pItem;

        reverse_iterator(VmaRawList<T>* pList, VmaListItem<T>* pItem) : m_pList(pList),  m_pItem(pItem) {}
    };
    class const_iterator
    {
        friend class VmaList<T, AllocatorT>;
    public:
        const_iterator() : m_pList(VMA_NULL), m_pItem(VMA_NULL) {}
        const_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}
        const_iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        iterator drop_const() { return { const_cast<VmaRawList<T>*>(m_pList), const_cast<VmaListItem<T>*>(m_pItem) }; }

        const T& operator*() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return m_pItem->Value; }
        const T* operator->() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return &m_pItem->Value; }

        bool operator==(const const_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const const_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const_iterator operator++(int) { const_iterator result = *this; ++* this; return result; }
        const_iterator operator--(int) { const_iterator result = *this; --* this; return result; }

        const_iterator& operator++() { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); m_pItem = m_pItem->pNext; return *this; }
        const_iterator& operator--();

    private:
        const VmaRawList<T>* m_pList;
        const VmaListItem<T>* m_pItem;

        const_iterator(const VmaRawList<T>* pList, const VmaListItem<T>* pItem) : m_pList(pList), m_pItem(pItem) {}
    };
    class const_reverse_iterator
    {
        friend class VmaList<T, AllocatorT>;
    public:
        const_reverse_iterator() : m_pList(VMA_NULL), m_pItem(VMA_NULL) {}
        const_reverse_iterator(const reverse_iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}
        const_reverse_iterator(const iterator& src) : m_pList(src.m_pList), m_pItem(src.m_pItem) {}

        reverse_iterator drop_const() { return { const_cast<VmaRawList<T>*>(m_pList), const_cast<VmaListItem<T>*>(m_pItem) }; }

        const T& operator*() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return m_pItem->Value; }
        const T* operator->() const { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); return &m_pItem->Value; }

        bool operator==(const const_reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem == rhs.m_pItem; }
        bool operator!=(const const_reverse_iterator& rhs) const { VMA_HEAVY_ASSERT(m_pList == rhs.m_pList); return m_pItem != rhs.m_pItem; }

        const_reverse_iterator operator++(int) { const_reverse_iterator result = *this; ++* this; return result; }
        const_reverse_iterator operator--(int) { const_reverse_iterator result = *this; --* this; return result; }

        const_reverse_iterator& operator++() { VMA_HEAVY_ASSERT(m_pItem != VMA_NULL); m_pItem = m_pItem->pPrev; return *this; }
        const_reverse_iterator& operator--();

    private:
        const VmaRawList<T>* m_pList;
        const VmaListItem<T>* m_pItem;

        const_reverse_iterator(const VmaRawList<T>* pList, const VmaListItem<T>* pItem) : m_pList(pList), m_pItem(pItem) {}
    };

    VmaList(const AllocatorT& allocator) : m_RawList(allocator.m_pCallbacks) {}

    bool empty() const { return m_RawList.IsEmpty(); }
    size_t size() const { return m_RawList.GetCount(); }

    iterator begin() { return iterator(&m_RawList, m_RawList.Front()); }
    iterator end() { return iterator(&m_RawList, VMA_NULL); }

    const_iterator cbegin() const { return const_iterator(&m_RawList, m_RawList.Front()); }
    const_iterator cend() const { return const_iterator(&m_RawList, VMA_NULL); }

    const_iterator begin() const { return cbegin(); }
    const_iterator end() const { return cend(); }

    reverse_iterator rbegin() { return reverse_iterator(&m_RawList, m_RawList.Back()); }
    reverse_iterator rend() { return reverse_iterator(&m_RawList, VMA_NULL); }

    const_reverse_iterator crbegin() const { return const_reverse_iterator(&m_RawList, m_RawList.Back()); }
    const_reverse_iterator crend() const { return const_reverse_iterator(&m_RawList, VMA_NULL); }

    const_reverse_iterator rbegin() const { return crbegin(); }
    const_reverse_iterator rend() const { return crend(); }

    void push_back(const T& value) { m_RawList.PushBack(value); }
    iterator insert(iterator it, const T& value) { return iterator(&m_RawList, m_RawList.InsertBefore(it.m_pItem, value)); }

    void clear() { m_RawList.Clear(); }
    void erase(iterator it) { m_RawList.Remove(it.m_pItem); }

private:
    VmaRawList<T> m_RawList;
};

#ifndef _VMA_LIST_FUNCTIONS
template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::iterator& VmaList<T, AllocatorT>::iterator::operator--()
{
    if (m_pItem != VMA_NULL)
    {
        m_pItem = m_pItem->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::reverse_iterator& VmaList<T, AllocatorT>::reverse_iterator::operator--()
{
    if (m_pItem != VMA_NULL)
    {
        m_pItem = m_pItem->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Front();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::const_iterator& VmaList<T, AllocatorT>::const_iterator::operator--()
{
    if (m_pItem != VMA_NULL)
    {
        m_pItem = m_pItem->pPrev;
    }
    else
    {
        VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}

template<typename T, typename AllocatorT>
typename VmaList<T, AllocatorT>::const_reverse_iterator& VmaList<T, AllocatorT>::const_reverse_iterator::operator--()
{
    if (m_pItem != VMA_NULL)
    {
        m_pItem = m_pItem->pNext;
    }
    else
    {
        VMA_HEAVY_ASSERT(!m_pList->IsEmpty());
        m_pItem = m_pList->Back();
    }
    return *this;
}
#endif // _VMA_LIST_FUNCTIONS
#endif // _VMA_LIST

#ifndef _VMA_INTRUSIVE_LINKED_LIST
/*
Expected interface of ItemTypeTraits:
struct MyItemTypeTraits
{
    typedef MyItem ItemType;
    static ItemType* GetPrev(const ItemType* item) { return item->myPrevPtr; }
    static ItemType* GetNext(const ItemType* item) { return item->myNextPtr; }
    static ItemType*& AccessPrev(ItemType* item) { return item->myPrevPtr; }
    static ItemType*& AccessNext(ItemType* item) { return item->myNextPtr; }
};
*/
template<typename ItemTypeTraits>
class VmaIntrusiveLinkedList
{
public:
    typedef typename ItemTypeTraits::ItemType ItemType;
    static ItemType* GetPrev(const ItemType* item) { return ItemTypeTraits::GetPrev(item); }
    static ItemType* GetNext(const ItemType* item) { return ItemTypeTraits::GetNext(item); }

    // Movable, not copyable.
    VmaIntrusiveLinkedList() = default;
    VmaIntrusiveLinkedList(VmaIntrusiveLinkedList && src);
    VmaIntrusiveLinkedList(const VmaIntrusiveLinkedList&) = delete;
    VmaIntrusiveLinkedList& operator=(VmaIntrusiveLinkedList&& src);
    VmaIntrusiveLinkedList& operator=(const VmaIntrusiveLinkedList&) = delete;
    ~VmaIntrusiveLinkedList() { VMA_HEAVY_ASSERT(IsEmpty()); }

    size_t GetCount() const { return m_Count; }
    bool IsEmpty() const { return m_Count == 0; }
    ItemType* Front() { return m_Front; }
    ItemType* Back() { return m_Back; }
    const ItemType* Front() const { return m_Front; }
    const ItemType* Back() const { return m_Back; }

    void PushBack(ItemType* item);
    void PushFront(ItemType* item);
    ItemType* PopBack();
    ItemType* PopFront();

    // MyItem can be null - it means PushBack.
    void InsertBefore(ItemType* existingItem, ItemType* newItem);
    // MyItem can be null - it means PushFront.
    void InsertAfter(ItemType* existingItem, ItemType* newItem);
    void Remove(ItemType* item);
    void RemoveAll();

private:
    ItemType* m_Front = VMA_NULL;
    ItemType* m_Back = VMA_NULL;
    size_t m_Count = 0;
};

#ifndef _VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
template<typename ItemTypeTraits>
VmaIntrusiveLinkedList<ItemTypeTraits>::VmaIntrusiveLinkedList(VmaIntrusiveLinkedList&& src)
    : m_Front(src.m_Front), m_Back(src.m_Back), m_Count(src.m_Count)
{
    src.m_Front = src.m_Back = VMA_NULL;
    src.m_Count = 0;
}

template<typename ItemTypeTraits>
VmaIntrusiveLinkedList<ItemTypeTraits>& VmaIntrusiveLinkedList<ItemTypeTraits>::operator=(VmaIntrusiveLinkedList&& src)
{
    if (&src != this)
    {
        VMA_HEAVY_ASSERT(IsEmpty());
        m_Front = src.m_Front;
        m_Back = src.m_Back;
        m_Count = src.m_Count;
        src.m_Front = src.m_Back = VMA_NULL;
        src.m_Count = 0;
    }
    return *this;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::PushBack(ItemType* item)
{
    VMA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == VMA_NULL && ItemTypeTraits::GetNext(item) == VMA_NULL);
    if (IsEmpty())
    {
        m_Front = item;
        m_Back = item;
        m_Count = 1;
    }
    else
    {
        ItemTypeTraits::AccessPrev(item) = m_Back;
        ItemTypeTraits::AccessNext(m_Back) = item;
        m_Back = item;
        ++m_Count;
    }
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::PushFront(ItemType* item)
{
    VMA_HEAVY_ASSERT(ItemTypeTraits::GetPrev(item) == VMA_NULL && ItemTypeTraits::GetNext(item) == VMA_NULL);
    if (IsEmpty())
    {
        m_Front = item;
        m_Back = item;
        m_Count = 1;
    }
    else
    {
        ItemTypeTraits::AccessNext(item) = m_Front;
        ItemTypeTraits::AccessPrev(m_Front) = item;
        m_Front = item;
        ++m_Count;
    }
}

template<typename ItemTypeTraits>
typename VmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* VmaIntrusiveLinkedList<ItemTypeTraits>::PopBack()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const backItem = m_Back;
    ItemType* const prevItem = ItemTypeTraits::GetPrev(backItem);
    if (prevItem != VMA_NULL)
    {
        ItemTypeTraits::AccessNext(prevItem) = VMA_NULL;
    }
    m_Back = prevItem;
    --m_Count;
    ItemTypeTraits::AccessPrev(backItem) = VMA_NULL;
    ItemTypeTraits::AccessNext(backItem) = VMA_NULL;
    return backItem;
}

template<typename ItemTypeTraits>
typename VmaIntrusiveLinkedList<ItemTypeTraits>::ItemType* VmaIntrusiveLinkedList<ItemTypeTraits>::PopFront()
{
    VMA_HEAVY_ASSERT(m_Count > 0);
    ItemType* const frontItem = m_Front;
    ItemType* const nextItem = ItemTypeTraits::GetNext(frontItem);
    if (nextItem != VMA_NULL)
    {
        ItemTypeTraits::AccessPrev(nextItem) = VMA_NULL;
    }
    m_Front = nextItem;
    --m_Count;
    ItemTypeTraits::AccessPrev(frontItem) = VMA_NULL;
    ItemTypeTraits::AccessNext(frontItem) = VMA_NULL;
    return frontItem;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::InsertBefore(ItemType* existingItem, ItemType* newItem)
{
    VMA_HEAVY_ASSERT(newItem != VMA_NULL && ItemTypeTraits::GetPrev(newItem) == VMA_NULL && ItemTypeTraits::GetNext(newItem) == VMA_NULL);
    if (existingItem != VMA_NULL)
    {
        ItemType* const prevItem = ItemTypeTraits::GetPrev(existingItem);
        ItemTypeTraits::AccessPrev(newItem) = prevItem;
        ItemTypeTraits::AccessNext(newItem) = existingItem;
        ItemTypeTraits::AccessPrev(existingItem) = newItem;
        if (prevItem != VMA_NULL)
        {
            ItemTypeTraits::AccessNext(prevItem) = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_Front == existingItem);
            m_Front = newItem;
        }
        ++m_Count;
    }
    else
        PushBack(newItem);
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::InsertAfter(ItemType* existingItem, ItemType* newItem)
{
    VMA_HEAVY_ASSERT(newItem != VMA_NULL && ItemTypeTraits::GetPrev(newItem) == VMA_NULL && ItemTypeTraits::GetNext(newItem) == VMA_NULL);
    if (existingItem != VMA_NULL)
    {
        ItemType* const nextItem = ItemTypeTraits::GetNext(existingItem);
        ItemTypeTraits::AccessNext(newItem) = nextItem;
        ItemTypeTraits::AccessPrev(newItem) = existingItem;
        ItemTypeTraits::AccessNext(existingItem) = newItem;
        if (nextItem != VMA_NULL)
        {
            ItemTypeTraits::AccessPrev(nextItem) = newItem;
        }
        else
        {
            VMA_HEAVY_ASSERT(m_Back == existingItem);
            m_Back = newItem;
        }
        ++m_Count;
    }
    else
        return PushFront(newItem);
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::Remove(ItemType* item)
{
    VMA_HEAVY_ASSERT(item != VMA_NULL && m_Count > 0);
    if (ItemTypeTraits::GetPrev(item) != VMA_NULL)
    {
        ItemTypeTraits::AccessNext(ItemTypeTraits::AccessPrev(item)) = ItemTypeTraits::GetNext(item);
    }
    else
    {
        VMA_HEAVY_ASSERT(m_Front == item);
        m_Front = ItemTypeTraits::GetNext(item);
    }

    if (ItemTypeTraits::GetNext(item) != VMA_NULL)
    {
        ItemTypeTraits::AccessPrev(ItemTypeTraits::AccessNext(item)) = ItemTypeTraits::GetPrev(item);
    }
    else
    {
        VMA_HEAVY_ASSERT(m_Back == item);
        m_Back = ItemTypeTraits::GetPrev(item);
    }
    ItemTypeTraits::AccessPrev(item) = VMA_NULL;
    ItemTypeTraits::AccessNext(item) = VMA_NULL;
    --m_Count;
}

template<typename ItemTypeTraits>
void VmaIntrusiveLinkedList<ItemTypeTraits>::RemoveAll()
{
    if (!IsEmpty())
    {
        ItemType* item = m_Back;
        while (item != VMA_NULL)
        {
            ItemType* const prevItem = ItemTypeTraits::AccessPrev(item);
            ItemTypeTraits::AccessPrev(item) = VMA_NULL;
            ItemTypeTraits::AccessNext(item) = VMA_NULL;
            item = prevItem;
        }
        m_Front = VMA_NULL;
        m_Back = VMA_NULL;
        m_Count = 0;
    }
}
#endif // _VMA_INTRUSIVE_LINKED_LIST_FUNCTIONS
#endif // _VMA_INTRUSIVE_LINKED_LIST

// Unused in this version.
#if 0

#ifndef _VMA_PAIR
template<typename T1, typename T2>
struct VmaPair
{
    T1 first;
    T2 second;

    VmaPair() : first(), second() {}
    VmaPair(const T1& firstSrc, const T2& secondSrc) : first(firstSrc), second(secondSrc) {}
};

template<typename FirstT, typename SecondT>
struct VmaPairFirstLess
{
    bool operator()(const VmaPair<FirstT, SecondT>& lhs, const VmaPair<FirstT, SecondT>& rhs) const
    {
        return lhs.first < rhs.first;
    }
    bool operator()(const VmaPair<FirstT, SecondT>& lhs, const FirstT& rhsFirst) const
    {
        return lhs.first < rhsFirst;
    }
};
#endif // _VMA_PAIR

#ifndef _VMA_MAP
/* Class compatible with subset of interface of std::unordered_map.
KeyT, ValueT must be POD because they will be stored in VmaVector.
*/
template<typename KeyT, typename ValueT>
class VmaMap
{
public:
    typedef VmaPair<KeyT, ValueT> PairType;
    typedef PairType* iterator;

    VmaMap(const VmaStlAllocator<PairType>& allocator) : m_Vector(allocator) {}

    iterator begin() { return m_Vector.begin(); }
    iterator end() { return m_Vector.end(); }
    size_t size() { return m_Vector.size(); }

    void insert(const PairType& pair);
    iterator find(const KeyT& key);
    void erase(iterator it);

private:
    VmaVector< PairType, VmaStlAllocator<PairType>> m_Vector;
};

#ifndef _VMA_MAP_FUNCTIONS
template<typename KeyT, typename ValueT>
void VmaMap<KeyT, ValueT>::insert(const PairType& pair)
{
    const size_t indexToInsert = VmaBinaryFindFirstNotLess(
        m_Vector.data(),
        m_Vector.data() + m_Vector.size(),
        pair,
        VmaPairFirstLess<KeyT, ValueT>()) - m_Vector.data();
    VmaVectorInsert(m_Vector, indexToInsert, pair);
}

template<typename KeyT, typename ValueT>
VmaPair<KeyT, ValueT>* VmaMap<KeyT, ValueT>::find(const KeyT& key)
{
    PairType* it = VmaBinaryFindFirstNotLess(
        m_Vector.data(),
        m_Vector.data() + m_Vector.size(),
        key,
        VmaPairFirstLess<KeyT, ValueT>());
    if ((it != m_Vector.end()) && (it->first == key))
    {
        return it;
    }
    else
    {
        return m_Vector.end();
    }
}

template<typename KeyT, typename ValueT>
void VmaMap<KeyT, ValueT>::erase(iterator it)
{
    VmaVectorRemove(m_Vector, it - m_Vector.begin());
}
#endif // _VMA_MAP_FUNCTIONS
#endif // _VMA_MAP

#endif // #if 0

#if !defined(_VMA_STRING_BUILDER) && VMA_STATS_STRING_ENABLED
class VmaStringBuilder
{
public:
    VmaStringBuilder(const VkAllocationCallbacks* allocationCallbacks) : m_Data(VmaStlAllocator<char>(allocationCallbacks)) {}
    ~VmaStringBuilder() = default;

    size_t GetLength() const { return m_Data.size(); }
    const char* GetData() const { return m_Data.data(); }
    void AddNewLine() { Add('\n'); }
    void Add(char ch) { m_Data.push_back(ch); }

    void Add(const char* pStr);
    void AddNumber(uint32_t num);
    void AddNumber(uint64_t num);
    void AddPointer(const void* ptr);

private:
    VmaVector<char, VmaStlAllocator<char>> m_Data;
};

#ifndef _VMA_STRING_BUILDER_FUNCTIONS
void VmaStringBuilder::Add(const char* pStr)
{
    const size_t strLen = strlen(pStr);
    if (strLen > 0)
    {
        const size_t oldCount = m_Data.size();
        m_Data.resize(oldCount + strLen);
        memcpy(m_Data.data() + oldCount, pStr, strLen);
    }
}

void VmaStringBuilder::AddNumber(uint32_t num)
{
    char buf[11];
    buf[10] = '\0';
    char* p = &buf[10];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    Add(p);
}

void VmaStringBuilder::AddNumber(uint64_t num)
{
    char buf[21];
    buf[20] = '\0';
    char* p = &buf[20];
    do
    {
        *--p = '0' + (char)(num % 10);
        num /= 10;
    } while (num);
    Add(p);
}

void VmaStringBuilder::AddPointer(const void* ptr)
{
    char buf[21];
    VmaPtrToStr(buf, sizeof(buf), ptr);
    Add(buf);
}
#endif //_VMA_STRING_BUILDER_FUNCTIONS
#endif // _VMA_STRING_BUILDER

#if !defined(_VMA_JSON_WRITER) && VMA_STATS_STRING_ENABLED
/*
Allows to conveniently build a correct JSON document to be written to the
VmaStringBuilder passed to the constructor.
*/
class VmaJsonWriter
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaJsonWriter)
public:
    // sb - string builder to write the document to. Must remain alive for the whole lifetime of this object.
    VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb);
    ~VmaJsonWriter();

    // Begins object by writing "{".
    // Inside an object, you must call pairs of WriteString and a value, e.g.:
    // j.BeginObject(true); j.WriteString("A"); j.WriteNumber(1); j.WriteString("B"); j.WriteNumber(2); j.EndObject();
    // Will write: { "A": 1, "B": 2 }
    void BeginObject(bool singleLine = false);
    // Ends object by writing "}".
    void EndObject();

    // Begins array by writing "[".
    // Inside an array, you can write a sequence of any values.
    void BeginArray(bool singleLine = false);
    // Ends array by writing "[".
    void EndArray();

    // Writes a string value inside "".
    // pStr can contain any ANSI characters, including '"', new line etc. - they will be properly escaped.
    void WriteString(const char* pStr);

    // Begins writing a string value.
    // Call BeginString, ContinueString, ContinueString, ..., EndString instead of
    // WriteString to conveniently build the string content incrementally, made of
    // parts including numbers.
    void BeginString(const char* pStr = VMA_NULL);
    // Posts next part of an open string.
    void ContinueString(const char* pStr);
    // Posts next part of an open string. The number is converted to decimal characters.
    void ContinueString(uint32_t n);
    void ContinueString(uint64_t n);
    // Posts next part of an open string. Pointer value is converted to characters
    // using "%p" formatting - shown as hexadecimal number, e.g.: 000000081276Ad00
    void ContinueString_Pointer(const void* ptr);
    // Ends writing a string value by writing '"'.
    void EndString(const char* pStr = VMA_NULL);

    // Writes a number value.
    void WriteNumber(uint32_t n);
    void WriteNumber(uint64_t n);
    // Writes a boolean value - false or true.
    void WriteBool(bool b);
    // Writes a null value.
    void WriteNull();

private:
    enum COLLECTION_TYPE
    {
        COLLECTION_TYPE_OBJECT,
        COLLECTION_TYPE_ARRAY,
    };
    struct StackItem
    {
        COLLECTION_TYPE type;
        uint32_t valueCount;
        bool singleLineMode;
    };

    static const char* const INDENT;

    VmaStringBuilder& m_SB;
    VmaVector< StackItem, VmaStlAllocator<StackItem> > m_Stack;
    bool m_InsideString;

    void BeginValue(bool isString);
    void WriteIndent(bool oneLess = false);
};
const char* const VmaJsonWriter::INDENT = "  ";

#ifndef _VMA_JSON_WRITER_FUNCTIONS
VmaJsonWriter::VmaJsonWriter(const VkAllocationCallbacks* pAllocationCallbacks, VmaStringBuilder& sb)
    : m_SB(sb),
    m_Stack(VmaStlAllocator<StackItem>(pAllocationCallbacks)),
    m_InsideString(false) {}

VmaJsonWriter::~VmaJsonWriter()
{
    VMA_ASSERT(!m_InsideString);
    VMA_ASSERT(m_Stack.empty());
}

void VmaJsonWriter::BeginObject(bool singleLine)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('{');

    StackItem item;
    item.type = COLLECTION_TYPE_OBJECT;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void VmaJsonWriter::EndObject()
{
    VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add('}');

    VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_OBJECT);
    m_Stack.pop_back();
}

void VmaJsonWriter::BeginArray(bool singleLine)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(false);
    m_SB.Add('[');

    StackItem item;
    item.type = COLLECTION_TYPE_ARRAY;
    item.valueCount = 0;
    item.singleLineMode = singleLine;
    m_Stack.push_back(item);
}

void VmaJsonWriter::EndArray()
{
    VMA_ASSERT(!m_InsideString);

    WriteIndent(true);
    m_SB.Add(']');

    VMA_ASSERT(!m_Stack.empty() && m_Stack.back().type == COLLECTION_TYPE_ARRAY);
    m_Stack.pop_back();
}

void VmaJsonWriter::WriteString(const char* pStr)
{
    BeginString(pStr);
    EndString();
}

void VmaJsonWriter::BeginString(const char* pStr)
{
    VMA_ASSERT(!m_InsideString);

    BeginValue(true);
    m_SB.Add('"');
    m_InsideString = true;
    if (pStr != VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
}

void VmaJsonWriter::ContinueString(const char* pStr)
{
    VMA_ASSERT(m_InsideString);

    const size_t strLen = strlen(pStr);
    for (size_t i = 0; i < strLen; ++i)
    {
        char ch = pStr[i];
        if (ch == '\\')
        {
            m_SB.Add("\\\\");
        }
        else if (ch == '"')
        {
            m_SB.Add("\\\"");
        }
        else if (ch >= 32)
        {
            m_SB.Add(ch);
        }
        else switch (ch)
        {
        case '\b':
            m_SB.Add("\\b");
            break;
        case '\f':
            m_SB.Add("\\f");
            break;
        case '\n':
            m_SB.Add("\\n");
            break;
        case '\r':
            m_SB.Add("\\r");
            break;
        case '\t':
            m_SB.Add("\\t");
            break;
        default:
            VMA_ASSERT(0 && "Character not currently supported.");
        }
    }
}

void VmaJsonWriter::ContinueString(uint32_t n)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::ContinueString(uint64_t n)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::ContinueString_Pointer(const void* ptr)
{
    VMA_ASSERT(m_InsideString);
    m_SB.AddPointer(ptr);
}

void VmaJsonWriter::EndString(const char* pStr)
{
    VMA_ASSERT(m_InsideString);
    if (pStr != VMA_NULL && pStr[0] != '\0')
    {
        ContinueString(pStr);
    }
    m_SB.Add('"');
    m_InsideString = false;
}

void VmaJsonWriter::WriteNumber(uint32_t n)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::WriteNumber(uint64_t n)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.AddNumber(n);
}

void VmaJsonWriter::WriteBool(bool b)
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add(b ? "true" : "false");
}

void VmaJsonWriter::WriteNull()
{
    VMA_ASSERT(!m_InsideString);
    BeginValue(false);
    m_SB.Add("null");
}

void VmaJsonWriter::BeginValue(bool isString)
{
    if (!m_Stack.empty())
    {
        StackItem& currItem = m_Stack.back();
        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 == 0)
        {
            VMA_ASSERT(isString);
        }

        if (currItem.type == COLLECTION_TYPE_OBJECT &&
            currItem.valueCount % 2 != 0)
        {
            m_SB.Add(": ");
        }
        else if (currItem.valueCount > 0)
        {
            m_SB.Add(", ");
            WriteIndent();
        }
        else
        {
            WriteIndent();
        }
        ++currItem.valueCount;
    }
}

void VmaJsonWriter::WriteIndent(bool oneLess)
{
    if (!m_Stack.empty() && !m_Stack.back().singleLineMode)
    {
        m_SB.AddNewLine();

        size_t count = m_Stack.size();
        if (count > 0 && oneLess)
        {
            --count;
        }
        for (size_t i = 0; i < count; ++i)
        {
            m_SB.Add(INDENT);
        }
    }
}
#endif // _VMA_JSON_WRITER_FUNCTIONS

static void VmaPrintDetailedStatistics(VmaJsonWriter& json, const VmaDetailedStatistics& stat)
{
    json.BeginObject();

    json.WriteString("BlockCount");
    json.WriteNumber(stat.statistics.blockCount);
    json.WriteString("BlockBytes");
    json.WriteNumber(stat.statistics.blockBytes);
    json.WriteString("AllocationCount");
    json.WriteNumber(stat.statistics.allocationCount);
    json.WriteString("AllocationBytes");
    json.WriteNumber(stat.statistics.allocationBytes);
    json.WriteString("UnusedRangeCount");
    json.WriteNumber(stat.unusedRangeCount);

    if (stat.statistics.allocationCount > 1)
    {
        json.WriteString("AllocationSizeMin");
        json.WriteNumber(stat.allocationSizeMin);
        json.WriteString("AllocationSizeMax");
        json.WriteNumber(stat.allocationSizeMax);
    }
    if (stat.unusedRangeCount > 1)
    {
        json.WriteString("UnusedRangeSizeMin");
        json.WriteNumber(stat.unusedRangeSizeMin);
        json.WriteString("UnusedRangeSizeMax");
        json.WriteNumber(stat.unusedRangeSizeMax);
    }
    json.EndObject();
}
#endif // _VMA_JSON_WRITER

#ifndef _VMA_MAPPING_HYSTERESIS

class VmaMappingHysteresis
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaMappingHysteresis)
public:
    VmaMappingHysteresis() = default;

    uint32_t GetExtraMapping() const { return m_ExtraMapping; }

    // Call when Map was called.
    // Returns true if switched to extra +1 mapping reference count.
    bool PostMap()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 0)
        {
            ++m_MajorCounter;
            if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING)
            {
                m_ExtraMapping = 1;
                m_MajorCounter = 0;
                m_MinorCounter = 0;
                return true;
            }
        }
        else // m_ExtraMapping == 1
            PostMinorCounter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

    // Call when Unmap was called.
    void PostUnmap()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 0)
            ++m_MajorCounter;
        else // m_ExtraMapping == 1
            PostMinorCounter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was made from the memory block.
    void PostAlloc()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 1)
            ++m_MajorCounter;
        else // m_ExtraMapping == 0
            PostMinorCounter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
    }

    // Call when allocation was freed from the memory block.
    // Returns true if switched to extra -1 mapping reference count.
    bool PostFree()
    {
#if VMA_MAPPING_HYSTERESIS_ENABLED
        if(m_ExtraMapping == 1)
        {
            ++m_MajorCounter;
            if(m_MajorCounter >= COUNTER_MIN_EXTRA_MAPPING &&
                m_MajorCounter > m_MinorCounter + 1)
            {
                m_ExtraMapping = 0;
                m_MajorCounter = 0;
                m_MinorCounter = 0;
                return true;
            }
        }
        else // m_ExtraMapping == 0
            PostMinorCounter();
#endif // #if VMA_MAPPING_HYSTERESIS_ENABLED
        return false;
    }

private:
    static const int32_t COUNTER_MIN_EXTRA_MAPPING = 7;

    uint32_t m_MinorCounter = 0;
    uint32_t m_MajorCounter = 0;
    uint32_t m_ExtraMapping = 0; // 0 or 1.

    void PostMinorCounter()
    {
        if(m_MinorCounter < m_MajorCounter)
        {
            ++m_MinorCounter;
        }
        else if(m_MajorCounter > 0)
        {
            --m_MajorCounter;
            --m_MinorCounter;
        }
    }
};

#endif // _VMA_MAPPING_HYSTERESIS

#ifndef _VMA_DEVICE_MEMORY_BLOCK
/*
Represents a single block of device memory (`VkDeviceMemory`) with all the
data about its regions (aka suballocations, #VmaAllocation), assigned and free.

Thread-safety:
- Access to m_pMetadata must be externally synchronized.
- Map, Unmap, Bind* are synchronized internally.
*/
class VmaDeviceMemoryBlock
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDeviceMemoryBlock)
public:
    VmaBlockMetadata* m_pMetadata;

    VmaDeviceMemoryBlock(VmaAllocator hAllocator);
    ~VmaDeviceMemoryBlock();

    // Always call after construction.
    void Init(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t newMemoryTypeIndex,
        VkDeviceMemory newMemory,
        VkDeviceSize newSize,
        uint32_t id,
        uint32_t algorithm,
        VkDeviceSize bufferImageGranularity);
    // Always call before destruction.
    void Destroy(VmaAllocator allocator);

    VmaPool GetParentPool() const { return m_hParentPool; }
    VkDeviceMemory GetDeviceMemory() const { return m_hMemory; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    uint32_t GetId() const { return m_Id; }
    void* GetMappedData() const { return m_pMappedData; }
    uint32_t GetMapRefCount() const { return m_MapCount; }

    // Call when allocation/free was made from m_pMetadata.
    // Used for m_MappingHysteresis.
    void PostAlloc(VmaAllocator hAllocator);
    void PostFree(VmaAllocator hAllocator);

    // Validates all data structures inside this object. If not valid, returns false.
    bool Validate() const;
    VkResult CheckCorruption(VmaAllocator hAllocator);

    // ppData can be null.
    VkResult Map(VmaAllocator hAllocator, uint32_t count, void** ppData);
    void Unmap(VmaAllocator hAllocator, uint32_t count);

    VkResult WriteMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);
    VkResult ValidateMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize);

    VkResult BindBufferMemory(
        const VmaAllocator hAllocator,
        const VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        const VmaAllocator hAllocator,
        const VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

private:
    VmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
    uint32_t m_MemoryTypeIndex;
    uint32_t m_Id;
    VkDeviceMemory m_hMemory;

    /*
    Protects access to m_hMemory so it is not used by multiple threads simultaneously, e.g. vkMapMemory, vkBindBufferMemory.
    Also protects m_MapCount, m_pMappedData.
    Allocations, deallocations, any change in m_pMetadata is protected by parent's VmaBlockVector::m_Mutex.
    */
    VMA_MUTEX m_MapAndBindMutex;
    VmaMappingHysteresis m_MappingHysteresis;
    uint32_t m_MapCount;
    void* m_pMappedData;
};
#endif // _VMA_DEVICE_MEMORY_BLOCK

#ifndef _VMA_ALLOCATION_T
struct VmaAllocation_T
{
    friend struct VmaDedicatedAllocationListItemTraits;

    enum FLAGS
    {
        FLAG_PERSISTENT_MAP   = 0x01,
        FLAG_MAPPING_ALLOWED  = 0x02,
    };

public:
    enum ALLOCATION_TYPE
    {
        ALLOCATION_TYPE_NONE,
        ALLOCATION_TYPE_BLOCK,
        ALLOCATION_TYPE_DEDICATED,
    };

    // This struct is allocated using VmaPoolAllocator.
    VmaAllocation_T(bool mappingAllowed);
    ~VmaAllocation_T();

    void InitBlockAllocation(
        VmaDeviceMemoryBlock* block,
        VmaAllocHandle allocHandle,
        VkDeviceSize alignment,
        VkDeviceSize size,
        uint32_t memoryTypeIndex,
        VmaSuballocationType suballocationType,
        bool mapped);
    // pMappedData not null means allocation is created with MAPPED flag.
    void InitDedicatedAllocation(
        VmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceMemory hMemory,
        VmaSuballocationType suballocationType,
        void* pMappedData,
        VkDeviceSize size);

    ALLOCATION_TYPE GetType() const { return (ALLOCATION_TYPE)m_Type; }
    VkDeviceSize GetAlignment() const { return m_Alignment; }
    VkDeviceSize GetSize() const { return m_Size; }
    void* GetUserData() const { return m_pUserData; }
    const char* GetName() const { return m_pName; }
    VmaSuballocationType GetSuballocationType() const { return (VmaSuballocationType)m_SuballocationType; }

    VmaDeviceMemoryBlock* GetBlock() const { VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK); return m_BlockAllocation.m_Block; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    bool IsPersistentMap() const { return (m_Flags & FLAG_PERSISTENT_MAP) != 0; }
    bool IsMappingAllowed() const { return (m_Flags & FLAG_MAPPING_ALLOWED) != 0; }

    void SetUserData(VmaAllocator hAllocator, void* pUserData) { m_pUserData = pUserData; }
    void SetName(VmaAllocator hAllocator, const char* pName);
    void FreeName(VmaAllocator hAllocator);
    uint8_t SwapBlockAllocation(VmaAllocator hAllocator, VmaAllocation allocation);
    VmaAllocHandle GetAllocHandle() const;
    VkDeviceSize GetOffset() const;
    VmaPool GetParentPool() const;
    VkDeviceMemory GetMemory() const;
    void* GetMappedData() const;

    void BlockAllocMap();
    void BlockAllocUnmap();
    VkResult DedicatedAllocMap(VmaAllocator hAllocator, void** ppData);
    void DedicatedAllocUnmap(VmaAllocator hAllocator);

#if VMA_STATS_STRING_ENABLED
    uint32_t GetBufferImageUsage() const { return m_BufferImageUsage; }

    void InitBufferImageUsage(uint32_t bufferImageUsage);
    void PrintParameters(class VmaJsonWriter& json) const;
#endif

private:
    // Allocation out of VmaDeviceMemoryBlock.
    struct BlockAllocation
    {
        VmaDeviceMemoryBlock* m_Block;
        VmaAllocHandle m_AllocHandle;
    };
    // Allocation for an object that has its own private VkDeviceMemory.
    struct DedicatedAllocation
    {
        VmaPool m_hParentPool; // VK_NULL_HANDLE if not belongs to custom pool.
        VkDeviceMemory m_hMemory;
        void* m_pMappedData; // Not null means memory is mapped.
        VmaAllocation_T* m_Prev;
        VmaAllocation_T* m_Next;
    };
    union
    {
        // Allocation out of VmaDeviceMemoryBlock.
        BlockAllocation m_BlockAllocation;
        // Allocation for an object that has its own private VkDeviceMemory.
        DedicatedAllocation m_DedicatedAllocation;
    };

    VkDeviceSize m_Alignment;
    VkDeviceSize m_Size;
    void* m_pUserData;
    char* m_pName;
    uint32_t m_MemoryTypeIndex;
    uint8_t m_Type; // ALLOCATION_TYPE
    uint8_t m_SuballocationType; // VmaSuballocationType
    // Reference counter for vmaMapMemory()/vmaUnmapMemory().
    uint8_t m_MapCount;
    uint8_t m_Flags; // enum FLAGS
#if VMA_STATS_STRING_ENABLED
    uint32_t m_BufferImageUsage; // 0 if unknown.
#endif
};
#endif // _VMA_ALLOCATION_T

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS
struct VmaDedicatedAllocationListItemTraits
{
    typedef VmaAllocation_T ItemType;

    static ItemType* GetPrev(const ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Prev;
    }
    static ItemType* GetNext(const ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Next;
    }
    static ItemType*& AccessPrev(ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Prev;
    }
    static ItemType*& AccessNext(ItemType* item)
    {
        VMA_HEAVY_ASSERT(item->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);
        return item->m_DedicatedAllocation.m_Next;
    }
};
#endif // _VMA_DEDICATED_ALLOCATION_LIST_ITEM_TRAITS

#ifndef _VMA_DEDICATED_ALLOCATION_LIST
/*
Stores linked list of VmaAllocation_T objects.
Thread-safe, synchronized internally.
*/
class VmaDedicatedAllocationList
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDedicatedAllocationList)
public:
    VmaDedicatedAllocationList() {}
    ~VmaDedicatedAllocationList();

    void Init(bool useMutex) { m_UseMutex = useMutex; }
    bool Validate();

    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats);
    void AddStatistics(VmaStatistics& inoutStats);
#if VMA_STATS_STRING_ENABLED
    // Writes JSON array with the list of allocations.
    void BuildStatsString(VmaJsonWriter& json);
#endif

    bool IsEmpty();
    void Register(VmaAllocation alloc);
    void Unregister(VmaAllocation alloc);

private:
    typedef VmaIntrusiveLinkedList<VmaDedicatedAllocationListItemTraits> DedicatedAllocationLinkedList;

    bool m_UseMutex = true;
    VMA_RW_MUTEX m_Mutex;
    DedicatedAllocationLinkedList m_AllocationList;
};

#ifndef _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS

VmaDedicatedAllocationList::~VmaDedicatedAllocationList()
{
    VMA_HEAVY_ASSERT(Validate());

    if (!m_AllocationList.IsEmpty())
    {
        VMA_ASSERT(false && "Unfreed dedicated allocations found!");
    }
}

bool VmaDedicatedAllocationList::Validate()
{
    const size_t declaredCount = m_AllocationList.GetCount();
    size_t actualCount = 0;
    VmaMutexLockRead lock(m_Mutex, m_UseMutex);
    for (VmaAllocation alloc = m_AllocationList.Front();
        alloc != VMA_NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        ++actualCount;
    }
    VMA_VALIDATE(actualCount == declaredCount);

    return true;
}

void VmaDedicatedAllocationList::AddDetailedStatistics(VmaDetailedStatistics& inoutStats)
{
    for(auto* item = m_AllocationList.Front(); item != nullptr; item = DedicatedAllocationLinkedList::GetNext(item))
    {
        const VkDeviceSize size = item->GetSize();
        inoutStats.statistics.blockCount++;
        inoutStats.statistics.blockBytes += size;
        VmaAddDetailedStatisticsAllocation(inoutStats, item->GetSize());
    }
}

void VmaDedicatedAllocationList::AddStatistics(VmaStatistics& inoutStats)
{
    VmaMutexLockRead lock(m_Mutex, m_UseMutex);

    const uint32_t allocCount = (uint32_t)m_AllocationList.GetCount();
    inoutStats.blockCount += allocCount;
    inoutStats.allocationCount += allocCount;

    for(auto* item = m_AllocationList.Front(); item != nullptr; item = DedicatedAllocationLinkedList::GetNext(item))
    {
        const VkDeviceSize size = item->GetSize();
        inoutStats.blockBytes += size;
        inoutStats.allocationBytes += size;
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaDedicatedAllocationList::BuildStatsString(VmaJsonWriter& json)
{
    VmaMutexLockRead lock(m_Mutex, m_UseMutex);
    json.BeginArray();
    for (VmaAllocation alloc = m_AllocationList.Front();
        alloc != VMA_NULL; alloc = m_AllocationList.GetNext(alloc))
    {
        json.BeginObject(true);
        alloc->PrintParameters(json);
        json.EndObject();
    }
    json.EndArray();
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaDedicatedAllocationList::IsEmpty()
{
    VmaMutexLockRead lock(m_Mutex, m_UseMutex);
    return m_AllocationList.IsEmpty();
}

void VmaDedicatedAllocationList::Register(VmaAllocation alloc)
{
    VmaMutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.PushBack(alloc);
}

void VmaDedicatedAllocationList::Unregister(VmaAllocation alloc)
{
    VmaMutexLockWrite lock(m_Mutex, m_UseMutex);
    m_AllocationList.Remove(alloc);
}
#endif // _VMA_DEDICATED_ALLOCATION_LIST_FUNCTIONS
#endif // _VMA_DEDICATED_ALLOCATION_LIST

#ifndef _VMA_SUBALLOCATION
/*
Represents a region of VmaDeviceMemoryBlock that is either assigned and returned as
allocated memory block or free.
*/
struct VmaSuballocation
{
    VkDeviceSize offset;
    VkDeviceSize size;
    void* userData;
    VmaSuballocationType type;
};

// Comparator for offsets.
struct VmaSuballocationOffsetLess
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset < rhs.offset;
    }
};

struct VmaSuballocationOffsetGreater
{
    bool operator()(const VmaSuballocation& lhs, const VmaSuballocation& rhs) const
    {
        return lhs.offset > rhs.offset;
    }
};

struct VmaSuballocationItemSizeLess
{
    bool operator()(const VmaSuballocationList::iterator lhs,
        const VmaSuballocationList::iterator rhs) const
    {
        return lhs->size < rhs->size;
    }

    bool operator()(const VmaSuballocationList::iterator lhs,
        VkDeviceSize rhsSize) const
    {
        return lhs->size < rhsSize;
    }
};
#endif // _VMA_SUBALLOCATION

#ifndef _VMA_ALLOCATION_REQUEST
/*
Parameters of planned allocation inside a VmaDeviceMemoryBlock.
item points to a FREE suballocation.
*/
struct VmaAllocationRequest
{
    VmaAllocHandle allocHandle;
    VkDeviceSize size;
    VmaSuballocationList::iterator item;
    void* customData;
    uint64_t algorithmData;
    VmaAllocationRequestType type;
};
#endif // _VMA_ALLOCATION_REQUEST

#ifndef _VMA_BLOCK_METADATA
/*
Data structure used for bookkeeping of allocations and unused ranges of memory
in a single VkDeviceMemory block.
*/
class VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata)
public:
    // pAllocationCallbacks, if not null, must be owned externally - alive and unchanged for the whole lifetime of this object.
    VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata() = default;

    virtual void Init(VkDeviceSize size) { m_Size = size; }
    bool IsVirtual() const { return m_IsVirtual; }
    VkDeviceSize GetSize() const { return m_Size; }

    // Validates all data structures inside this object. If not valid, returns false.
    virtual bool Validate() const = 0;
    virtual size_t GetAllocationCount() const = 0;
    virtual size_t GetFreeRegionsCount() const = 0;
    virtual VkDeviceSize GetSumFreeSize() const = 0;
    // Returns true if this block is empty - contains only single free suballocation.
    virtual bool IsEmpty() const = 0;
    virtual void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) = 0;
    virtual VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const = 0;
    virtual void* GetAllocationUserData(VmaAllocHandle allocHandle) const = 0;

    virtual VmaAllocHandle GetAllocationListBegin() const = 0;
    virtual VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const = 0;
    virtual VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const = 0;

    // Shouldn't modify blockCount.
    virtual void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const = 0;
    virtual void AddStatistics(VmaStatistics& inoutStats) const = 0;

#if VMA_STATS_STRING_ENABLED
    virtual void PrintDetailedMap(class VmaJsonWriter& json) const = 0;
#endif

    // Tries to find a place for suballocation with given parameters inside this block.
    // If succeeded, fills pAllocationRequest and returns true.
    // If failed, returns false.
    virtual bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        // Always one of VMA_ALLOCATION_CREATE_STRATEGY_* or VMA_ALLOCATION_INTERNAL_STRATEGY_* flags.
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) = 0;

    virtual VkResult CheckCorruption(const void* pBlockData) = 0;

    // Makes actual allocation based on request. Request must already be checked and valid.
    virtual void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) = 0;

    // Frees suballocation assigned to given memory region.
    virtual void Free(VmaAllocHandle allocHandle) = 0;

    // Frees all allocations.
    // Careful! Don't call it if there are VmaAllocation objects owned by userData of cleared allocations!
    virtual void Clear() = 0;

    virtual void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) = 0;
    virtual void DebugLogAllAllocations() const = 0;

protected:
    const VkAllocationCallbacks* GetAllocationCallbacks() const { return m_pAllocationCallbacks; }
    VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
    VkDeviceSize GetDebugMargin() const { return VkDeviceSize(IsVirtual() ? 0 : VMA_DEBUG_MARGIN); }

    void DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const;
#if VMA_STATS_STRING_ENABLED
    // mapRefCount == UINT32_MAX means unspecified.
    void PrintDetailedMap_Begin(class VmaJsonWriter& json,
        VkDeviceSize unusedBytes,
        size_t allocationCount,
        size_t unusedRangeCount) const;
    void PrintDetailedMap_Allocation(class VmaJsonWriter& json,
        VkDeviceSize offset, VkDeviceSize size, void* userData) const;
    void PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
        VkDeviceSize offset,
        VkDeviceSize size) const;
    void PrintDetailedMap_End(class VmaJsonWriter& json) const;
#endif

private:
    VkDeviceSize m_Size;
    const VkAllocationCallbacks* m_pAllocationCallbacks;
    const VkDeviceSize m_BufferImageGranularity;
    const bool m_IsVirtual;
};

#ifndef _VMA_BLOCK_METADATA_FUNCTIONS
VmaBlockMetadata::VmaBlockMetadata(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : m_Size(0),
    m_pAllocationCallbacks(pAllocationCallbacks),
    m_BufferImageGranularity(bufferImageGranularity),
    m_IsVirtual(isVirtual) {}

void VmaBlockMetadata::DebugLogAllocation(VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    if (IsVirtual())
    {
        VMA_DEBUG_LOG_FORMAT("UNFREED VIRTUAL ALLOCATION; Offset: %llu; Size: %llu; UserData: %p", offset, size, userData);
    }
    else
    {
        VMA_ASSERT(userData != VMA_NULL);
        VmaAllocation allocation = reinterpret_cast<VmaAllocation>(userData);

        userData = allocation->GetUserData();
        const char* name = allocation->GetName();

#if VMA_STATS_STRING_ENABLED
        VMA_DEBUG_LOG_FORMAT("UNFREED ALLOCATION; Offset: %llu; Size: %llu; UserData: %p; Name: %s; Type: %s; Usage: %u",
            offset, size, userData, name ? name : "vma_empty",
            VMA_SUBALLOCATION_TYPE_NAMES[allocation->GetSuballocationType()],
            allocation->GetBufferImageUsage());
#else
        VMA_DEBUG_LOG_FORMAT("UNFREED ALLOCATION; Offset: %llu; Size: %llu; UserData: %p; Name: %s; Type: %u",
            offset, size, userData, name ? name : "vma_empty",
            (uint32_t)allocation->GetSuballocationType());
#endif // VMA_STATS_STRING_ENABLED
    }

}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata::PrintDetailedMap_Begin(class VmaJsonWriter& json,
    VkDeviceSize unusedBytes, size_t allocationCount, size_t unusedRangeCount) const
{
    json.WriteString("TotalBytes");
    json.WriteNumber(GetSize());

    json.WriteString("UnusedBytes");
    json.WriteNumber(unusedBytes);

    json.WriteString("Allocations");
    json.WriteNumber((uint64_t)allocationCount);

    json.WriteString("UnusedRanges");
    json.WriteNumber((uint64_t)unusedRangeCount);

    json.WriteString("Suballocations");
    json.BeginArray();
}

void VmaBlockMetadata::PrintDetailedMap_Allocation(class VmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size, void* userData) const
{
    json.BeginObject(true);

    json.WriteString("Offset");
    json.WriteNumber(offset);

    if (IsVirtual())
    {
        json.WriteString("Size");
        json.WriteNumber(size);
        if (userData)
        {
            json.WriteString("CustomData");
            json.BeginString();
            json.ContinueString_Pointer(userData);
            json.EndString();
        }
    }
    else
    {
        ((VmaAllocation)userData)->PrintParameters(json);
    }

    json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_UnusedRange(class VmaJsonWriter& json,
    VkDeviceSize offset, VkDeviceSize size) const
{
    json.BeginObject(true);

    json.WriteString("Offset");
    json.WriteNumber(offset);

    json.WriteString("Type");
    json.WriteString(VMA_SUBALLOCATION_TYPE_NAMES[VMA_SUBALLOCATION_TYPE_FREE]);

    json.WriteString("Size");
    json.WriteNumber(size);

    json.EndObject();
}

void VmaBlockMetadata::PrintDetailedMap_End(class VmaJsonWriter& json) const
{
    json.EndArray();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_BLOCK_METADATA_FUNCTIONS
#endif // _VMA_BLOCK_METADATA

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY
// Before deleting object of this class remember to call 'Destroy()'
class VmaBlockBufferImageGranularity final
{
public:
    struct ValidationContext
    {
        const VkAllocationCallbacks* allocCallbacks;
        uint16_t* pageAllocs;
    };

    VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity);
    ~VmaBlockBufferImageGranularity();

    bool IsEnabled() const { return m_BufferImageGranularity > MAX_LOW_BUFFER_IMAGE_GRANULARITY; }

    void Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size);
    // Before destroying object you must call free it's memory
    void Destroy(const VkAllocationCallbacks* pAllocationCallbacks);

    void RoundupAllocRequest(VmaSuballocationType allocType,
        VkDeviceSize& inOutAllocSize,
        VkDeviceSize& inOutAllocAlignment) const;

    bool CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
        VkDeviceSize allocSize,
        VkDeviceSize blockOffset,
        VkDeviceSize blockSize,
        VmaSuballocationType allocType) const;

    void AllocPages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size);
    void FreePages(VkDeviceSize offset, VkDeviceSize size);
    void Clear();

    ValidationContext StartValidation(const VkAllocationCallbacks* pAllocationCallbacks,
        bool isVirutal) const;
    bool Validate(ValidationContext& ctx, VkDeviceSize offset, VkDeviceSize size) const;
    bool FinishValidation(ValidationContext& ctx) const;

private:
    static const uint16_t MAX_LOW_BUFFER_IMAGE_GRANULARITY = 256;

    struct RegionInfo
    {
        uint8_t allocType;
        uint16_t allocCount;
    };

    VkDeviceSize m_BufferImageGranularity;
    uint32_t m_RegionCount;
    RegionInfo* m_RegionInfo;

    uint32_t GetStartPage(VkDeviceSize offset) const { return OffsetToPageIndex(offset & ~(m_BufferImageGranularity - 1)); }
    uint32_t GetEndPage(VkDeviceSize offset, VkDeviceSize size) const { return OffsetToPageIndex((offset + size - 1) & ~(m_BufferImageGranularity - 1)); }

    uint32_t OffsetToPageIndex(VkDeviceSize offset) const;
    void AllocPage(RegionInfo& page, uint8_t allocType);
};

#ifndef _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
VmaBlockBufferImageGranularity::VmaBlockBufferImageGranularity(VkDeviceSize bufferImageGranularity)
    : m_BufferImageGranularity(bufferImageGranularity),
    m_RegionCount(0),
    m_RegionInfo(VMA_NULL) {}

VmaBlockBufferImageGranularity::~VmaBlockBufferImageGranularity()
{
    VMA_ASSERT(m_RegionInfo == VMA_NULL && "Free not called before destroying object!");
}

void VmaBlockBufferImageGranularity::Init(const VkAllocationCallbacks* pAllocationCallbacks, VkDeviceSize size)
{
    if (IsEnabled())
    {
        m_RegionCount = static_cast<uint32_t>(VmaDivideRoundingUp(size, m_BufferImageGranularity));
        m_RegionInfo = vma_new_array(pAllocationCallbacks, RegionInfo, m_RegionCount);
        memset(m_RegionInfo, 0, m_RegionCount * sizeof(RegionInfo));
    }
}

void VmaBlockBufferImageGranularity::Destroy(const VkAllocationCallbacks* pAllocationCallbacks)
{
    if (m_RegionInfo)
    {
        vma_delete_array(pAllocationCallbacks, m_RegionInfo, m_RegionCount);
        m_RegionInfo = VMA_NULL;
    }
}

void VmaBlockBufferImageGranularity::RoundupAllocRequest(VmaSuballocationType allocType,
    VkDeviceSize& inOutAllocSize,
    VkDeviceSize& inOutAllocAlignment) const
{
    if (m_BufferImageGranularity > 1 &&
        m_BufferImageGranularity <= MAX_LOW_BUFFER_IMAGE_GRANULARITY)
    {
        if (allocType == VMA_SUBALLOCATION_TYPE_UNKNOWN ||
            allocType == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
            allocType == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
        {
            inOutAllocAlignment = VMA_MAX(inOutAllocAlignment, m_BufferImageGranularity);
            inOutAllocSize = VmaAlignUp(inOutAllocSize, m_BufferImageGranularity);
        }
    }
}

bool VmaBlockBufferImageGranularity::CheckConflictAndAlignUp(VkDeviceSize& inOutAllocOffset,
    VkDeviceSize allocSize,
    VkDeviceSize blockOffset,
    VkDeviceSize blockSize,
    VmaSuballocationType allocType) const
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(inOutAllocOffset);
        if (m_RegionInfo[startPage].allocCount > 0 &&
            VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(m_RegionInfo[startPage].allocType), allocType))
        {
            inOutAllocOffset = VmaAlignUp(inOutAllocOffset, m_BufferImageGranularity);
            if (blockSize < allocSize + inOutAllocOffset - blockOffset)
                return true;
            ++startPage;
        }
        uint32_t endPage = GetEndPage(inOutAllocOffset, allocSize);
        if (endPage != startPage &&
            m_RegionInfo[endPage].allocCount > 0 &&
            VmaIsBufferImageGranularityConflict(static_cast<VmaSuballocationType>(m_RegionInfo[endPage].allocType), allocType))
        {
            return true;
        }
    }
    return false;
}

void VmaBlockBufferImageGranularity::AllocPages(uint8_t allocType, VkDeviceSize offset, VkDeviceSize size)
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(offset);
        AllocPage(m_RegionInfo[startPage], allocType);

        uint32_t endPage = GetEndPage(offset, size);
        if (startPage != endPage)
            AllocPage(m_RegionInfo[endPage], allocType);
    }
}

void VmaBlockBufferImageGranularity::FreePages(VkDeviceSize offset, VkDeviceSize size)
{
    if (IsEnabled())
    {
        uint32_t startPage = GetStartPage(offset);
        --m_RegionInfo[startPage].allocCount;
        if (m_RegionInfo[startPage].allocCount == 0)
            m_RegionInfo[startPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
        uint32_t endPage = GetEndPage(offset, size);
        if (startPage != endPage)
        {
            --m_RegionInfo[endPage].allocCount;
            if (m_RegionInfo[endPage].allocCount == 0)
                m_RegionInfo[endPage].allocType = VMA_SUBALLOCATION_TYPE_FREE;
        }
    }
}

void VmaBlockBufferImageGranularity::Clear()
{
    if (m_RegionInfo)
        memset(m_RegionInfo, 0, m_RegionCount * sizeof(RegionInfo));
}

VmaBlockBufferImageGranularity::ValidationContext VmaBlockBufferImageGranularity::StartValidation(
    const VkAllocationCallbacks* pAllocationCallbacks, bool isVirutal) const
{
    ValidationContext ctx{ pAllocationCallbacks, VMA_NULL };
    if (!isVirutal && IsEnabled())
    {
        ctx.pageAllocs = vma_new_array(pAllocationCallbacks, uint16_t, m_RegionCount);
        memset(ctx.pageAllocs, 0, m_RegionCount * sizeof(uint16_t));
    }
    return ctx;
}

bool VmaBlockBufferImageGranularity::Validate(ValidationContext& ctx,
    VkDeviceSize offset, VkDeviceSize size) const
{
    if (IsEnabled())
    {
        uint32_t start = GetStartPage(offset);
        ++ctx.pageAllocs[start];
        VMA_VALIDATE(m_RegionInfo[start].allocCount > 0);

        uint32_t end = GetEndPage(offset, size);
        if (start != end)
        {
            ++ctx.pageAllocs[end];
            VMA_VALIDATE(m_RegionInfo[end].allocCount > 0);
        }
    }
    return true;
}

bool VmaBlockBufferImageGranularity::FinishValidation(ValidationContext& ctx) const
{
    // Check proper page structure
    if (IsEnabled())
    {
        VMA_ASSERT(ctx.pageAllocs != VMA_NULL && "Validation context not initialized!");

        for (uint32_t page = 0; page < m_RegionCount; ++page)
        {
            VMA_VALIDATE(ctx.pageAllocs[page] == m_RegionInfo[page].allocCount);
        }
        vma_delete_array(ctx.allocCallbacks, ctx.pageAllocs, m_RegionCount);
        ctx.pageAllocs = VMA_NULL;
    }
    return true;
}

uint32_t VmaBlockBufferImageGranularity::OffsetToPageIndex(VkDeviceSize offset) const
{
    return static_cast<uint32_t>(offset >> VMA_BITSCAN_MSB(m_BufferImageGranularity));
}

void VmaBlockBufferImageGranularity::AllocPage(RegionInfo& page, uint8_t allocType)
{
    // When current alloc type is free then it can be overridden by new type
    if (page.allocCount == 0 || (page.allocCount > 0 && page.allocType == VMA_SUBALLOCATION_TYPE_FREE))
        page.allocType = allocType;

    ++page.allocCount;
}
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY_FUNCTIONS
#endif // _VMA_BLOCK_BUFFER_IMAGE_GRANULARITY

#if 0
#ifndef _VMA_BLOCK_METADATA_GENERIC
class VmaBlockMetadata_Generic : public VmaBlockMetadata
{
    friend class VmaDefragmentationAlgorithm_Generic;
    friend class VmaDefragmentationAlgorithm_Fast;
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_Generic)
public:
    VmaBlockMetadata_Generic(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata_Generic() = default;

    size_t GetAllocationCount() const override { return m_Suballocations.size() - m_FreeCount; }
    VkDeviceSize GetSumFreeSize() const override { return m_SumFreeSize; }
    bool IsEmpty() const override { return (m_Suballocations.size() == 1) && (m_FreeCount == 1); }
    void Free(VmaAllocHandle allocHandle) override { FreeSuballocation(FindAtOffset((VkDeviceSize)allocHandle - 1)); }
    VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;

    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json, uint32_t mapRefCount) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    VkResult CheckCorruption(const void* pBlockData) override;

    void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle GetAllocationListBegin() const override;
    VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
    void Clear() override;
    void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;
    void DebugLogAllAllocations() const override;

private:
    uint32_t m_FreeCount;
    VkDeviceSize m_SumFreeSize;
    VmaSuballocationList m_Suballocations;
    // Suballocations that are free. Sorted by size, ascending.
    VmaVector<VmaSuballocationList::iterator, VmaStlAllocator<VmaSuballocationList::iterator>> m_FreeSuballocationsBySize;

    VkDeviceSize AlignAllocationSize(VkDeviceSize size) const { return IsVirtual() ? size : VmaAlignUp(size, (VkDeviceSize)16); }

    VmaSuballocationList::iterator FindAtOffset(VkDeviceSize offset) const;
    bool ValidateFreeSuballocationList() const;

    // Checks if requested suballocation with given parameters can be placed in given pFreeSuballocItem.
    // If yes, fills pOffset and returns true. If no, returns false.
    bool CheckAllocation(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        VmaSuballocationList::const_iterator suballocItem,
        VmaAllocHandle* pAllocHandle) const;

    // Given free suballocation, it merges it with following one, which must also be free.
    void MergeFreeWithNext(VmaSuballocationList::iterator item);
    // Releases given suballocation, making it free.
    // Merges it with adjacent free suballocations if applicable.
    // Returns iterator to new free suballocation at this place.
    VmaSuballocationList::iterator FreeSuballocation(VmaSuballocationList::iterator suballocItem);
    // Given free suballocation, it inserts it into sorted list of
    // m_FreeSuballocationsBySize if it is suitable.
    void RegisterFreeSuballocation(VmaSuballocationList::iterator item);
    // Given free suballocation, it removes it from sorted list of
    // m_FreeSuballocationsBySize if it is suitable.
    void UnregisterFreeSuballocation(VmaSuballocationList::iterator item);
};

#ifndef _VMA_BLOCK_METADATA_GENERIC_FUNCTIONS
VmaBlockMetadata_Generic::VmaBlockMetadata_Generic(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_FreeCount(0),
    m_SumFreeSize(0),
    m_Suballocations(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
    m_FreeSuballocationsBySize(VmaStlAllocator<VmaSuballocationList::iterator>(pAllocationCallbacks)) {}

void VmaBlockMetadata_Generic::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);

    m_FreeCount = 1;
    m_SumFreeSize = size;

    VmaSuballocation suballoc = {};
    suballoc.offset = 0;
    suballoc.size = size;
    suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;

    m_Suballocations.push_back(suballoc);
    m_FreeSuballocationsBySize.push_back(m_Suballocations.begin());
}

bool VmaBlockMetadata_Generic::Validate() const
{
    VMA_VALIDATE(!m_Suballocations.empty());

    // Expected offset of new suballocation as calculated from previous ones.
    VkDeviceSize calculatedOffset = 0;
    // Expected number of free suballocations as calculated from traversing their list.
    uint32_t calculatedFreeCount = 0;
    // Expected sum size of free suballocations as calculated from traversing their list.
    VkDeviceSize calculatedSumFreeSize = 0;
    // Expected number of free suballocations that should be registered in
    // m_FreeSuballocationsBySize calculated from traversing their list.
    size_t freeSuballocationsToRegister = 0;
    // True if previous visited suballocation was free.
    bool prevFree = false;

    const VkDeviceSize debugMargin = GetDebugMargin();

    for (const auto& subAlloc : m_Suballocations)
    {
        // Actual offset of this suballocation doesn't match expected one.
        VMA_VALIDATE(subAlloc.offset == calculatedOffset);

        const bool currFree = (subAlloc.type == VMA_SUBALLOCATION_TYPE_FREE);
        // Two adjacent free suballocations are invalid. They should be merged.
        VMA_VALIDATE(!prevFree || !currFree);

        VmaAllocation alloc = (VmaAllocation)subAlloc.userData;
        if (!IsVirtual())
        {
            VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
        }

        if (currFree)
        {
            calculatedSumFreeSize += subAlloc.size;
            ++calculatedFreeCount;
            ++freeSuballocationsToRegister;

            // Margin required between allocations - every free space must be at least that large.
            VMA_VALIDATE(subAlloc.size >= debugMargin);
        }
        else
        {
            if (!IsVirtual())
            {
                VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == subAlloc.offset + 1);
                VMA_VALIDATE(alloc->GetSize() == subAlloc.size);
            }

            // Margin required between allocations - previous allocation must be free.
            VMA_VALIDATE(debugMargin == 0 || prevFree);
        }

        calculatedOffset += subAlloc.size;
        prevFree = currFree;
    }

    // Number of free suballocations registered in m_FreeSuballocationsBySize doesn't
    // match expected one.
    VMA_VALIDATE(m_FreeSuballocationsBySize.size() == freeSuballocationsToRegister);

    VkDeviceSize lastSize = 0;
    for (size_t i = 0; i < m_FreeSuballocationsBySize.size(); ++i)
    {
        VmaSuballocationList::iterator suballocItem = m_FreeSuballocationsBySize[i];

        // Only free suballocations can be registered in m_FreeSuballocationsBySize.
        VMA_VALIDATE(suballocItem->type == VMA_SUBALLOCATION_TYPE_FREE);
        // They must be sorted by size ascending.
        VMA_VALIDATE(suballocItem->size >= lastSize);

        lastSize = suballocItem->size;
    }

    // Check if totals match calculated values.
    VMA_VALIDATE(ValidateFreeSuballocationList());
    VMA_VALIDATE(calculatedOffset == GetSize());
    VMA_VALIDATE(calculatedSumFreeSize == m_SumFreeSize);
    VMA_VALIDATE(calculatedFreeCount == m_FreeCount);

    return true;
}

void VmaBlockMetadata_Generic::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
{
    const uint32_t rangeCount = (uint32_t)m_Suballocations.size();
    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += GetSize();

    for (const auto& suballoc : m_Suballocations)
    {
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
            VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);
        else
            VmaAddDetailedStatisticsUnusedRange(inoutStats, suballoc.size);
    }
}

void VmaBlockMetadata_Generic::AddStatistics(VmaStatistics& inoutStats) const
{
    inoutStats.blockCount++;
    inoutStats.allocationCount += (uint32_t)m_Suballocations.size() - m_FreeCount;
    inoutStats.blockBytes += GetSize();
    inoutStats.allocationBytes += GetSize() - m_SumFreeSize;
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Generic::PrintDetailedMap(class VmaJsonWriter& json, uint32_t mapRefCount) const
{
    PrintDetailedMap_Begin(json,
        m_SumFreeSize, // unusedBytes
        m_Suballocations.size() - (size_t)m_FreeCount, // allocationCount
        m_FreeCount, // unusedRangeCount
        mapRefCount);

    for (const auto& suballoc : m_Suballocations)
    {
        if (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            PrintDetailedMap_UnusedRange(json, suballoc.offset, suballoc.size);
        }
        else
        {
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);
        }
    }

    PrintDetailedMap_End(json);
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Generic::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(!upperAddress);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(pAllocationRequest != VMA_NULL);
    VMA_HEAVY_ASSERT(Validate());

    allocSize = AlignAllocationSize(allocSize);

    pAllocationRequest->type = VmaAllocationRequestType::Normal;
    pAllocationRequest->size = allocSize;

    const VkDeviceSize debugMargin = GetDebugMargin();

    // There is not enough total free space in this block to fulfill the request: Early return.
    if (m_SumFreeSize < allocSize + debugMargin)
    {
        return false;
    }

    // New algorithm, efficiently searching freeSuballocationsBySize.
    const size_t freeSuballocCount = m_FreeSuballocationsBySize.size();
    if (freeSuballocCount > 0)
    {
        if (strategy == 0 ||
            strategy == VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT)
        {
            // Find first free suballocation with size not less than allocSize + debugMargin.
            VmaSuballocationList::iterator* const it = VmaBinaryFindFirstNotLess(
                m_FreeSuballocationsBySize.data(),
                m_FreeSuballocationsBySize.data() + freeSuballocCount,
                allocSize + debugMargin,
                VmaSuballocationItemSizeLess());
            size_t index = it - m_FreeSuballocationsBySize.data();
            for (; index < freeSuballocCount; ++index)
            {
                if (CheckAllocation(
                    allocSize,
                    allocAlignment,
                    allocType,
                    m_FreeSuballocationsBySize[index],
                    &pAllocationRequest->allocHandle))
                {
                    pAllocationRequest->item = m_FreeSuballocationsBySize[index];
                    return true;
                }
            }
        }
        else if (strategy == VMA_ALLOCATION_INTERNAL_STRATEGY_MIN_OFFSET)
        {
            for (VmaSuballocationList::iterator it = m_Suballocations.begin();
                it != m_Suballocations.end();
                ++it)
            {
                if (it->type == VMA_SUBALLOCATION_TYPE_FREE && CheckAllocation(
                    allocSize,
                    allocAlignment,
                    allocType,
                    it,
                    &pAllocationRequest->allocHandle))
                {
                    pAllocationRequest->item = it;
                    return true;
                }
            }
        }
        else
        {
            VMA_ASSERT(strategy & (VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT | VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT ));
            // Search staring from biggest suballocations.
            for (size_t index = freeSuballocCount; index--; )
            {
                if (CheckAllocation(
                    allocSize,
                    allocAlignment,
                    allocType,
                    m_FreeSuballocationsBySize[index],
                    &pAllocationRequest->allocHandle))
                {
                    pAllocationRequest->item = m_FreeSuballocationsBySize[index];
                    return true;
                }
            }
        }
    }

    return false;
}

VkResult VmaBlockMetadata_Generic::CheckCorruption(const void* pBlockData)
{
    for (auto& suballoc : m_Suballocations)
    {
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_Generic::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::Normal);
    VMA_ASSERT(request.item != m_Suballocations.end());
    VmaSuballocation& suballoc = *request.item;
    // Given suballocation is a free block.
    VMA_ASSERT(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

    // Given offset is inside this suballocation.
    VMA_ASSERT((VkDeviceSize)request.allocHandle - 1 >= suballoc.offset);
    const VkDeviceSize paddingBegin = (VkDeviceSize)request.allocHandle - suballoc.offset - 1;
    VMA_ASSERT(suballoc.size >= paddingBegin + request.size);
    const VkDeviceSize paddingEnd = suballoc.size - paddingBegin - request.size;

    // Unregister this free suballocation from m_FreeSuballocationsBySize and update
    // it to become used.
    UnregisterFreeSuballocation(request.item);

    suballoc.offset = (VkDeviceSize)request.allocHandle - 1;
    suballoc.size = request.size;
    suballoc.type = type;
    suballoc.userData = userData;

    // If there are any free bytes remaining at the end, insert new free suballocation after current one.
    if (paddingEnd)
    {
        VmaSuballocation paddingSuballoc = {};
        paddingSuballoc.offset = suballoc.offset + suballoc.size;
        paddingSuballoc.size = paddingEnd;
        paddingSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
        VmaSuballocationList::iterator next = request.item;
        ++next;
        const VmaSuballocationList::iterator paddingEndItem =
            m_Suballocations.insert(next, paddingSuballoc);
        RegisterFreeSuballocation(paddingEndItem);
    }

    // If there are any free bytes remaining at the beginning, insert new free suballocation before current one.
    if (paddingBegin)
    {
        VmaSuballocation paddingSuballoc = {};
        paddingSuballoc.offset = suballoc.offset - paddingBegin;
        paddingSuballoc.size = paddingBegin;
        paddingSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
        const VmaSuballocationList::iterator paddingBeginItem =
            m_Suballocations.insert(request.item, paddingSuballoc);
        RegisterFreeSuballocation(paddingBeginItem);
    }

    // Update totals.
    m_FreeCount = m_FreeCount - 1;
    if (paddingBegin > 0)
    {
        ++m_FreeCount;
    }
    if (paddingEnd > 0)
    {
        ++m_FreeCount;
    }
    m_SumFreeSize -= request.size;
}

void VmaBlockMetadata_Generic::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    outInfo.offset = (VkDeviceSize)allocHandle - 1;
    const VmaSuballocation& suballoc = *FindAtOffset(outInfo.offset);
    outInfo.size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

void* VmaBlockMetadata_Generic::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
    return FindAtOffset((VkDeviceSize)allocHandle - 1)->userData;
}

VmaAllocHandle VmaBlockMetadata_Generic::GetAllocationListBegin() const
{
    if (IsEmpty())
        return VK_NULL_HANDLE;

    for (const auto& suballoc : m_Suballocations)
    {
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
            return (VmaAllocHandle)(suballoc.offset + 1);
    }
    VMA_ASSERT(false && "Should contain at least 1 allocation!");
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_Generic::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
    VmaSuballocationList::const_iterator prev = FindAtOffset((VkDeviceSize)prevAlloc - 1);

    for (VmaSuballocationList::const_iterator it = ++prev; it != m_Suballocations.end(); ++it)
    {
        if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
            return (VmaAllocHandle)(it->offset + 1);
    }
    return VK_NULL_HANDLE;
}

void VmaBlockMetadata_Generic::Clear()
{
    const VkDeviceSize size = GetSize();

    VMA_ASSERT(IsVirtual());
    m_FreeCount = 1;
    m_SumFreeSize = size;
    m_Suballocations.clear();
    m_FreeSuballocationsBySize.clear();

    VmaSuballocation suballoc = {};
    suballoc.offset = 0;
    suballoc.size = size;
    suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
    m_Suballocations.push_back(suballoc);

    m_FreeSuballocationsBySize.push_back(m_Suballocations.begin());
}

void VmaBlockMetadata_Generic::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
    VmaSuballocation& suballoc = *FindAtOffset((VkDeviceSize)allocHandle - 1);
    suballoc.userData = userData;
}

void VmaBlockMetadata_Generic::DebugLogAllAllocations() const
{
    for (const auto& suballoc : m_Suballocations)
    {
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
            DebugLogAllocation(suballoc.offset, suballoc.size, suballoc.userData);
    }
}

VmaSuballocationList::iterator VmaBlockMetadata_Generic::FindAtOffset(VkDeviceSize offset) const
{
    VMA_HEAVY_ASSERT(!m_Suballocations.empty());
    const VkDeviceSize last = m_Suballocations.rbegin()->offset;
    if (last == offset)
        return m_Suballocations.rbegin().drop_const();
    const VkDeviceSize first = m_Suballocations.begin()->offset;
    if (first == offset)
        return m_Suballocations.begin().drop_const();

    const size_t suballocCount = m_Suballocations.size();
    const VkDeviceSize step = (last - first + m_Suballocations.begin()->size) / suballocCount;
    auto findSuballocation = [&](auto begin, auto end) -> VmaSuballocationList::iterator
    {
        for (auto suballocItem = begin;
            suballocItem != end;
            ++suballocItem)
        {
            if (suballocItem->offset == offset)
                return suballocItem.drop_const();
        }
        VMA_ASSERT(false && "Not found!");
        return m_Suballocations.end().drop_const();
    };
    // If requested offset is closer to the end of range, search from the end
    if (offset - first > suballocCount * step / 2)
    {
        return findSuballocation(m_Suballocations.rbegin(), m_Suballocations.rend());
    }
    return findSuballocation(m_Suballocations.begin(), m_Suballocations.end());
}

bool VmaBlockMetadata_Generic::ValidateFreeSuballocationList() const
{
    VkDeviceSize lastSize = 0;
    for (size_t i = 0, count = m_FreeSuballocationsBySize.size(); i < count; ++i)
    {
        const VmaSuballocationList::iterator it = m_FreeSuballocationsBySize[i];

        VMA_VALIDATE(it->type == VMA_SUBALLOCATION_TYPE_FREE);
        VMA_VALIDATE(it->size >= lastSize);
        lastSize = it->size;
    }
    return true;
}

bool VmaBlockMetadata_Generic::CheckAllocation(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    VmaSuballocationList::const_iterator suballocItem,
    VmaAllocHandle* pAllocHandle) const
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(suballocItem != m_Suballocations.cend());
    VMA_ASSERT(pAllocHandle != VMA_NULL);

    const VkDeviceSize debugMargin = GetDebugMargin();
    const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();

    const VmaSuballocation& suballoc = *suballocItem;
    VMA_ASSERT(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

    // Size of this suballocation is too small for this request: Early return.
    if (suballoc.size < allocSize)
    {
        return false;
    }

    // Start from offset equal to beginning of this suballocation.
    VkDeviceSize offset = suballoc.offset + (suballocItem == m_Suballocations.cbegin() ? 0 : GetDebugMargin());

    // Apply debugMargin from the end of previous alloc.
    if (debugMargin > 0)
    {
        offset += debugMargin;
    }

    // Apply alignment.
    offset = VmaAlignUp(offset, allocAlignment);

    // Check previous suballocations for BufferImageGranularity conflicts.
    // Make bigger alignment if necessary.
    if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment)
    {
        bool bufferImageGranularityConflict = false;
        VmaSuballocationList::const_iterator prevSuballocItem = suballocItem;
        while (prevSuballocItem != m_Suballocations.cbegin())
        {
            --prevSuballocItem;
            const VmaSuballocation& prevSuballoc = *prevSuballocItem;
            if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, offset, bufferImageGranularity))
            {
                if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                {
                    bufferImageGranularityConflict = true;
                    break;
                }
            }
            else
                // Already on previous page.
                break;
        }
        if (bufferImageGranularityConflict)
        {
            offset = VmaAlignUp(offset, bufferImageGranularity);
        }
    }

    // Calculate padding at the beginning based on current offset.
    const VkDeviceSize paddingBegin = offset - suballoc.offset;

    // Fail if requested size plus margin after is bigger than size of this suballocation.
    if (paddingBegin + allocSize + debugMargin > suballoc.size)
    {
        return false;
    }

    // Check next suballocations for BufferImageGranularity conflicts.
    // If conflict exists, allocation cannot be made here.
    if (allocSize % bufferImageGranularity || offset % bufferImageGranularity)
    {
        VmaSuballocationList::const_iterator nextSuballocItem = suballocItem;
        ++nextSuballocItem;
        while (nextSuballocItem != m_Suballocations.cend())
        {
            const VmaSuballocation& nextSuballoc = *nextSuballocItem;
            if (VmaBlocksOnSamePage(offset, allocSize, nextSuballoc.offset, bufferImageGranularity))
            {
                if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                {
                    return false;
                }
            }
            else
            {
                // Already on next page.
                break;
            }
            ++nextSuballocItem;
        }
    }

    *pAllocHandle = (VmaAllocHandle)(offset + 1);
    // All tests passed: Success. pAllocHandle is already filled.
    return true;
}

void VmaBlockMetadata_Generic::MergeFreeWithNext(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item != m_Suballocations.end());
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);

    VmaSuballocationList::iterator nextItem = item;
    ++nextItem;
    VMA_ASSERT(nextItem != m_Suballocations.end());
    VMA_ASSERT(nextItem->type == VMA_SUBALLOCATION_TYPE_FREE);

    item->size += nextItem->size;
    --m_FreeCount;
    m_Suballocations.erase(nextItem);
}

VmaSuballocationList::iterator VmaBlockMetadata_Generic::FreeSuballocation(VmaSuballocationList::iterator suballocItem)
{
    // Change this suballocation to be marked as free.
    VmaSuballocation& suballoc = *suballocItem;
    suballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
    suballoc.userData = VMA_NULL;

    // Update totals.
    ++m_FreeCount;
    m_SumFreeSize += suballoc.size;

    // Merge with previous and/or next suballocation if it's also free.
    bool mergeWithNext = false;
    bool mergeWithPrev = false;

    VmaSuballocationList::iterator nextItem = suballocItem;
    ++nextItem;
    if ((nextItem != m_Suballocations.end()) && (nextItem->type == VMA_SUBALLOCATION_TYPE_FREE))
    {
        mergeWithNext = true;
    }

    VmaSuballocationList::iterator prevItem = suballocItem;
    if (suballocItem != m_Suballocations.begin())
    {
        --prevItem;
        if (prevItem->type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            mergeWithPrev = true;
        }
    }

    if (mergeWithNext)
    {
        UnregisterFreeSuballocation(nextItem);
        MergeFreeWithNext(suballocItem);
    }

    if (mergeWithPrev)
    {
        UnregisterFreeSuballocation(prevItem);
        MergeFreeWithNext(prevItem);
        RegisterFreeSuballocation(prevItem);
        return prevItem;
    }
    else
    {
        RegisterFreeSuballocation(suballocItem);
        return suballocItem;
    }
}

void VmaBlockMetadata_Generic::RegisterFreeSuballocation(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    if (m_FreeSuballocationsBySize.empty())
    {
        m_FreeSuballocationsBySize.push_back(item);
    }
    else
    {
        VmaVectorInsertSorted<VmaSuballocationItemSizeLess>(m_FreeSuballocationsBySize, item);
    }

    //VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}

void VmaBlockMetadata_Generic::UnregisterFreeSuballocation(VmaSuballocationList::iterator item)
{
    VMA_ASSERT(item->type == VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(item->size > 0);

    // You may want to enable this validation at the beginning or at the end of
    // this function, depending on what do you want to check.
    VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());

    VmaSuballocationList::iterator* const it = VmaBinaryFindFirstNotLess(
        m_FreeSuballocationsBySize.data(),
        m_FreeSuballocationsBySize.data() + m_FreeSuballocationsBySize.size(),
        item,
        VmaSuballocationItemSizeLess());
    for (size_t index = it - m_FreeSuballocationsBySize.data();
        index < m_FreeSuballocationsBySize.size();
        ++index)
    {
        if (m_FreeSuballocationsBySize[index] == item)
        {
            VmaVectorRemove(m_FreeSuballocationsBySize, index);
            return;
        }
        VMA_ASSERT((m_FreeSuballocationsBySize[index]->size == item->size) && "Not found.");
    }
    VMA_ASSERT(0 && "Not found.");

    //VMA_HEAVY_ASSERT(ValidateFreeSuballocationList());
}
#endif // _VMA_BLOCK_METADATA_GENERIC_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_GENERIC
#endif // #if 0

#ifndef _VMA_BLOCK_METADATA_LINEAR
/*
Allocations and their references in internal data structure look like this:

if(m_2ndVectorMode == SECOND_VECTOR_EMPTY):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER):

        0 +-------+
          | Alloc |  2nd[0]
          +-------+
          | Alloc |  2nd[1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
GetSize() +-------+

if(m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK):

        0 +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount]
          +-------+
          | Alloc |  1st[m_1stNullItemsBeginCount + 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  1st[1st.size() - 1]
          +-------+
          |       |
          |       |
          |       |
          +-------+
          | Alloc |  2nd[2nd.size() - 1]
          +-------+
          |  ...  |
          +-------+
          | Alloc |  2nd[1]
          +-------+
          | Alloc |  2nd[0]
GetSize() +-------+

*/
class VmaBlockMetadata_Linear : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_Linear)
public:
    VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata_Linear() = default;

    VkDeviceSize GetSumFreeSize() const override { return m_SumFreeSize; }
    bool IsEmpty() const override { return GetAllocationCount() == 0; }
    VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;
    size_t GetAllocationCount() const override;
    size_t GetFreeRegionsCount() const override;

    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    VkResult CheckCorruption(const void* pBlockData) override;

    void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void Free(VmaAllocHandle allocHandle) override;
    void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle GetAllocationListBegin() const override;
    VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
    VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const override;
    void Clear() override;
    void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;
    void DebugLogAllAllocations() const override;

private:
    /*
    There are two suballocation vectors, used in ping-pong way.
    The one with index m_1stVectorIndex is called 1st.
    The one with index (m_1stVectorIndex ^ 1) is called 2nd.
    2nd can be non-empty only when 1st is not empty.
    When 2nd is not empty, m_2ndVectorMode indicates its mode of operation.
    */
    typedef VmaVector<VmaSuballocation, VmaStlAllocator<VmaSuballocation>> SuballocationVectorType;

    enum SECOND_VECTOR_MODE
    {
        SECOND_VECTOR_EMPTY,
        /*
        Suballocations in 2nd vector are created later than the ones in 1st, but they
        all have smaller offset.
        */
        SECOND_VECTOR_RING_BUFFER,
        /*
        Suballocations in 2nd vector are upper side of double stack.
        They all have offsets higher than those in 1st vector.
        Top of this stack means smaller offsets, but higher indices in this vector.
        */
        SECOND_VECTOR_DOUBLE_STACK,
    };

    VkDeviceSize m_SumFreeSize;
    SuballocationVectorType m_Suballocations0, m_Suballocations1;
    uint32_t m_1stVectorIndex;
    SECOND_VECTOR_MODE m_2ndVectorMode;
    // Number of items in 1st vector with hAllocation = null at the beginning.
    size_t m_1stNullItemsBeginCount;
    // Number of other items in 1st vector with hAllocation = null somewhere in the middle.
    size_t m_1stNullItemsMiddleCount;
    // Number of items in 2nd vector with hAllocation = null.
    size_t m_2ndNullItemsCount;

    SuballocationVectorType& AccessSuballocations1st() { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    SuballocationVectorType& AccessSuballocations2nd() { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }
    const SuballocationVectorType& AccessSuballocations1st() const { return m_1stVectorIndex ? m_Suballocations1 : m_Suballocations0; }
    const SuballocationVectorType& AccessSuballocations2nd() const { return m_1stVectorIndex ? m_Suballocations0 : m_Suballocations1; }

    VmaSuballocation& FindSuballocation(VkDeviceSize offset) const;
    bool ShouldCompact1st() const;
    void CleanupAfterFree();

    bool CreateAllocationRequest_LowerAddress(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
    bool CreateAllocationRequest_UpperAddress(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
VmaBlockMetadata_Linear::VmaBlockMetadata_Linear(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_SumFreeSize(0),
    m_Suballocations0(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
    m_Suballocations1(VmaStlAllocator<VmaSuballocation>(pAllocationCallbacks)),
    m_1stVectorIndex(0),
    m_2ndVectorMode(SECOND_VECTOR_EMPTY),
    m_1stNullItemsBeginCount(0),
    m_1stNullItemsMiddleCount(0),
    m_2ndNullItemsCount(0) {}

void VmaBlockMetadata_Linear::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);
    m_SumFreeSize = size;
}

bool VmaBlockMetadata_Linear::Validate() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    VMA_VALIDATE(suballocations2nd.empty() == (m_2ndVectorMode == SECOND_VECTOR_EMPTY));
    VMA_VALIDATE(!suballocations1st.empty() ||
        suballocations2nd.empty() ||
        m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER);

    if (!suballocations1st.empty())
    {
        // Null item at the beginning should be accounted into m_1stNullItemsBeginCount.
        VMA_VALIDATE(suballocations1st[m_1stNullItemsBeginCount].type != VMA_SUBALLOCATION_TYPE_FREE);
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations1st.back().type != VMA_SUBALLOCATION_TYPE_FREE);
    }
    if (!suballocations2nd.empty())
    {
        // Null item at the end should be just pop_back().
        VMA_VALIDATE(suballocations2nd.back().type != VMA_SUBALLOCATION_TYPE_FREE);
    }

    VMA_VALIDATE(m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount <= suballocations1st.size());
    VMA_VALIDATE(m_2ndNullItemsCount <= suballocations2nd.size());

    VkDeviceSize sumUsedSize = 0;
    const size_t suballoc1stCount = suballocations1st.size();
    const VkDeviceSize debugMargin = GetDebugMargin();
    VkDeviceSize offset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = 0; i < suballoc2ndCount; ++i)
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
            if (!IsVirtual())
            {
                VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                    VMA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    for (size_t i = 0; i < m_1stNullItemsBeginCount; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        VMA_VALIDATE(suballoc.type == VMA_SUBALLOCATION_TYPE_FREE &&
            suballoc.userData == VMA_NULL);
    }

    size_t nullItem1stCount = m_1stNullItemsBeginCount;

    for (size_t i = m_1stNullItemsBeginCount; i < suballoc1stCount; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

        VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
        if (!IsVirtual())
        {
            VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
        }
        VMA_VALIDATE(suballoc.offset >= offset);
        VMA_VALIDATE(i >= m_1stNullItemsBeginCount || currFree);

        if (!currFree)
        {
            if (!IsVirtual())
            {
                VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                VMA_VALIDATE(alloc->GetSize() == suballoc.size);
            }
            sumUsedSize += suballoc.size;
        }
        else
        {
            ++nullItem1stCount;
        }

        offset = suballoc.offset + suballoc.size + debugMargin;
    }
    VMA_VALIDATE(nullItem1stCount == m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount);

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        const size_t suballoc2ndCount = suballocations2nd.size();
        size_t nullItem2ndCount = 0;
        for (size_t i = suballoc2ndCount; i--; )
        {
            const VmaSuballocation& suballoc = suballocations2nd[i];
            const bool currFree = (suballoc.type == VMA_SUBALLOCATION_TYPE_FREE);

            VmaAllocation const alloc = (VmaAllocation)suballoc.userData;
            if (!IsVirtual())
            {
                VMA_VALIDATE(currFree == (alloc == VK_NULL_HANDLE));
            }
            VMA_VALIDATE(suballoc.offset >= offset);

            if (!currFree)
            {
                if (!IsVirtual())
                {
                    VMA_VALIDATE((VkDeviceSize)alloc->GetAllocHandle() == suballoc.offset + 1);
                    VMA_VALIDATE(alloc->GetSize() == suballoc.size);
                }
                sumUsedSize += suballoc.size;
            }
            else
            {
                ++nullItem2ndCount;
            }

            offset = suballoc.offset + suballoc.size + debugMargin;
        }

        VMA_VALIDATE(nullItem2ndCount == m_2ndNullItemsCount);
    }

    VMA_VALIDATE(offset <= GetSize());
    VMA_VALIDATE(m_SumFreeSize == GetSize() - sumUsedSize);

    return true;
}

size_t VmaBlockMetadata_Linear::GetAllocationCount() const
{
    return AccessSuballocations1st().size() - m_1stNullItemsBeginCount - m_1stNullItemsMiddleCount +
        AccessSuballocations2nd().size() - m_2ndNullItemsCount;
}

size_t VmaBlockMetadata_Linear::GetFreeRegionsCount() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return SIZE_MAX;
}

void VmaBlockMetadata_Linear::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += size;

    VkDeviceSize lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // There is free space from lastOffset to freeSpace1stTo2ndEnd.
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAllocIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                VmaAddDetailedStatisticsAllocation(inoutStats, suballoc.size);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // There is free space from lastOffset to size.
                if (lastOffset < size)
                {
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    VmaAddDetailedStatisticsUnusedRange(inoutStats, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }
}

void VmaBlockMetadata_Linear::AddStatistics(VmaStatistics& inoutStats) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const VkDeviceSize size = GetSize();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    inoutStats.blockCount++;
    inoutStats.blockBytes += size;
    inoutStats.allocationBytes += size - m_SumFreeSize;

    VkDeviceSize lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = m_1stNullItemsBeginCount;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++inoutStats.allocationCount;

            // Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++inoutStats.allocationCount;

                // Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                // End of loop.
                lastOffset = size;
            }
        }
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Linear::PrintDetailedMap(class VmaJsonWriter& json) const
{
    const VkDeviceSize size = GetSize();
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    const size_t suballoc1stCount = suballocations1st.size();
    const size_t suballoc2ndCount = suballocations2nd.size();

    // FIRST PASS

    size_t unusedRangeCount = 0;
    VkDeviceSize usedBytes = 0;

    VkDeviceSize lastOffset = 0;

    size_t alloc2ndCount = 0;
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    size_t nextAlloc1stIndex = m_1stNullItemsBeginCount;
    size_t alloc1stCount = 0;
    const VkDeviceSize freeSpace1stTo2ndEnd =
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ? suballocations2nd.back().offset : size;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                ++unusedRangeCount;
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            ++alloc1stCount;
            usedBytes += suballoc.size;

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < size)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                ++unusedRangeCount;
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    ++unusedRangeCount;
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                ++alloc2ndCount;
                usedBytes += suballoc.size;

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    ++unusedRangeCount;
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    const VkDeviceSize unusedBytes = size - usedBytes;
    PrintDetailedMap_Begin(json, unusedBytes, alloc1stCount + alloc2ndCount, unusedRangeCount);

    // SECOND PASS
    lastOffset = 0;

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        const VkDeviceSize freeSpace2ndTo1stEnd = suballocations1st[m_1stNullItemsBeginCount].offset;
        size_t nextAlloc2ndIndex = 0;
        while (lastOffset < freeSpace2ndTo1stEnd)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex < suballoc2ndCount &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                ++nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex < suballoc2ndCount)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                ++nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < freeSpace2ndTo1stEnd)
                {
                    // There is free space from lastOffset to freeSpace2ndTo1stEnd.
                    const VkDeviceSize unusedRangeSize = freeSpace2ndTo1stEnd - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = freeSpace2ndTo1stEnd;
            }
        }
    }

    nextAlloc1stIndex = m_1stNullItemsBeginCount;
    while (lastOffset < freeSpace1stTo2ndEnd)
    {
        // Find next non-null allocation or move nextAllocIndex to the end.
        while (nextAlloc1stIndex < suballoc1stCount &&
            suballocations1st[nextAlloc1stIndex].userData == VMA_NULL)
        {
            ++nextAlloc1stIndex;
        }

        // Found non-null allocation.
        if (nextAlloc1stIndex < suballoc1stCount)
        {
            const VmaSuballocation& suballoc = suballocations1st[nextAlloc1stIndex];

            // 1. Process free space before this allocation.
            if (lastOffset < suballoc.offset)
            {
                // There is free space from lastOffset to suballoc.offset.
                const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // 2. Process this allocation.
            // There is allocation with suballoc.offset, suballoc.size.
            PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

            // 3. Prepare for next iteration.
            lastOffset = suballoc.offset + suballoc.size;
            ++nextAlloc1stIndex;
        }
        // We are at the end.
        else
        {
            if (lastOffset < freeSpace1stTo2ndEnd)
            {
                // There is free space from lastOffset to freeSpace1stTo2ndEnd.
                const VkDeviceSize unusedRangeSize = freeSpace1stTo2ndEnd - lastOffset;
                PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
            }

            // End of loop.
            lastOffset = freeSpace1stTo2ndEnd;
        }
    }

    if (m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        size_t nextAlloc2ndIndex = suballocations2nd.size() - 1;
        while (lastOffset < size)
        {
            // Find next non-null allocation or move nextAlloc2ndIndex to the end.
            while (nextAlloc2ndIndex != SIZE_MAX &&
                suballocations2nd[nextAlloc2ndIndex].userData == VMA_NULL)
            {
                --nextAlloc2ndIndex;
            }

            // Found non-null allocation.
            if (nextAlloc2ndIndex != SIZE_MAX)
            {
                const VmaSuballocation& suballoc = suballocations2nd[nextAlloc2ndIndex];

                // 1. Process free space before this allocation.
                if (lastOffset < suballoc.offset)
                {
                    // There is free space from lastOffset to suballoc.offset.
                    const VkDeviceSize unusedRangeSize = suballoc.offset - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // 2. Process this allocation.
                // There is allocation with suballoc.offset, suballoc.size.
                PrintDetailedMap_Allocation(json, suballoc.offset, suballoc.size, suballoc.userData);

                // 3. Prepare for next iteration.
                lastOffset = suballoc.offset + suballoc.size;
                --nextAlloc2ndIndex;
            }
            // We are at the end.
            else
            {
                if (lastOffset < size)
                {
                    // There is free space from lastOffset to size.
                    const VkDeviceSize unusedRangeSize = size - lastOffset;
                    PrintDetailedMap_UnusedRange(json, lastOffset, unusedRangeSize);
                }

                // End of loop.
                lastOffset = size;
            }
        }
    }

    PrintDetailedMap_End(json);
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Linear::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0);
    VMA_ASSERT(allocType != VMA_SUBALLOCATION_TYPE_FREE);
    VMA_ASSERT(pAllocationRequest != VMA_NULL);
    VMA_HEAVY_ASSERT(Validate());
    pAllocationRequest->size = allocSize;
    return upperAddress ?
        CreateAllocationRequest_UpperAddress(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest) :
        CreateAllocationRequest_LowerAddress(
            allocSize, allocAlignment, allocType, strategy, pAllocationRequest);
}

VkResult VmaBlockMetadata_Linear::CheckCorruption(const void* pBlockData)
{
    VMA_ASSERT(!IsVirtual());
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (size_t i = m_1stNullItemsBeginCount, count = suballocations1st.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations1st[i];
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (size_t i = 0, count = suballocations2nd.size(); i < count; ++i)
    {
        const VmaSuballocation& suballoc = suballocations2nd[i];
        if (suballoc.type != VMA_SUBALLOCATION_TYPE_FREE)
        {
            if (!VmaValidateMagicValue(pBlockData, suballoc.offset + suballoc.size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_Linear::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    const VkDeviceSize offset = (VkDeviceSize)request.allocHandle - 1;
    const VmaSuballocation newSuballoc = { offset, request.size, userData, type };

    switch (request.type)
    {
    case VmaAllocationRequestType::UpperAddress:
    {
        VMA_ASSERT(m_2ndVectorMode != SECOND_VECTOR_RING_BUFFER &&
            "CRITICAL ERROR: Trying to use linear allocator as double stack while it was already used as ring buffer.");
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
        suballocations2nd.push_back(newSuballoc);
        m_2ndVectorMode = SECOND_VECTOR_DOUBLE_STACK;
    }
    break;
    case VmaAllocationRequestType::EndOf1st:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();

        VMA_ASSERT(suballocations1st.empty() ||
            offset >= suballocations1st.back().offset + suballocations1st.back().size);
        // Check if it fits before the end of the block.
        VMA_ASSERT(offset + request.size <= GetSize());

        suballocations1st.push_back(newSuballoc);
    }
    break;
    case VmaAllocationRequestType::EndOf2nd:
    {
        SuballocationVectorType& suballocations1st = AccessSuballocations1st();
        // New allocation at the end of 2-part ring buffer, so before first allocation from 1st vector.
        VMA_ASSERT(!suballocations1st.empty() &&
            offset + request.size <= suballocations1st[m_1stNullItemsBeginCount].offset);
        SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

        switch (m_2ndVectorMode)
        {
        case SECOND_VECTOR_EMPTY:
            // First allocation from second part ring buffer.
            VMA_ASSERT(suballocations2nd.empty());
            m_2ndVectorMode = SECOND_VECTOR_RING_BUFFER;
            break;
        case SECOND_VECTOR_RING_BUFFER:
            // 2-part ring buffer is already started.
            VMA_ASSERT(!suballocations2nd.empty());
            break;
        case SECOND_VECTOR_DOUBLE_STACK:
            VMA_ASSERT(0 && "CRITICAL ERROR: Trying to use linear allocator as ring buffer while it was already used as double stack.");
            break;
        default:
            VMA_ASSERT(0);
        }

        suballocations2nd.push_back(newSuballoc);
    }
    break;
    default:
        VMA_ASSERT(0 && "CRITICAL INTERNAL ERROR.");
    }

    m_SumFreeSize -= newSuballoc.size;
}

void VmaBlockMetadata_Linear::Free(VmaAllocHandle allocHandle)
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    VkDeviceSize offset = (VkDeviceSize)allocHandle - 1;

    if (!suballocations1st.empty())
    {
        // First allocation: Mark it as next empty at the beginning.
        VmaSuballocation& firstSuballoc = suballocations1st[m_1stNullItemsBeginCount];
        if (firstSuballoc.offset == offset)
        {
            firstSuballoc.type = VMA_SUBALLOCATION_TYPE_FREE;
            firstSuballoc.userData = VMA_NULL;
            m_SumFreeSize += firstSuballoc.size;
            ++m_1stNullItemsBeginCount;
            CleanupAfterFree();
            return;
        }
    }

    // Last allocation in 2-part ring buffer or top of upper stack (same logic).
    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ||
        m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        VmaSuballocation& lastSuballoc = suballocations2nd.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations2nd.pop_back();
            CleanupAfterFree();
            return;
        }
    }
    // Last allocation in 1st vector.
    else if (m_2ndVectorMode == SECOND_VECTOR_EMPTY)
    {
        VmaSuballocation& lastSuballoc = suballocations1st.back();
        if (lastSuballoc.offset == offset)
        {
            m_SumFreeSize += lastSuballoc.size;
            suballocations1st.pop_back();
            CleanupAfterFree();
            return;
        }
    }

    VmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the middle of 1st vector.
    {
        const SuballocationVectorType::iterator it = VmaBinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            VmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = VMA_NULL;
            ++m_1stNullItemsMiddleCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Item from the middle of 2nd vector.
        const SuballocationVectorType::iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            it->type = VMA_SUBALLOCATION_TYPE_FREE;
            it->userData = VMA_NULL;
            ++m_2ndNullItemsCount;
            m_SumFreeSize += it->size;
            CleanupAfterFree();
            return;
        }
    }

    VMA_ASSERT(0 && "Allocation to free not found in linear allocator!");
}

void VmaBlockMetadata_Linear::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    outInfo.offset = (VkDeviceSize)allocHandle - 1;
    VmaSuballocation& suballoc = FindSuballocation(outInfo.offset);
    outInfo.size = suballoc.size;
    outInfo.pUserData = suballoc.userData;
}

void* VmaBlockMetadata_Linear::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
    return FindSuballocation((VkDeviceSize)allocHandle - 1).userData;
}

VmaAllocHandle VmaBlockMetadata_Linear::GetAllocationListBegin() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_Linear::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_Linear::GetNextFreeRegionSize(VmaAllocHandle alloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    VMA_ASSERT(0);
    return 0;
}

void VmaBlockMetadata_Linear::Clear()
{
    m_SumFreeSize = GetSize();
    m_Suballocations0.clear();
    m_Suballocations1.clear();
    // Leaving m_1stVectorIndex unchanged - it doesn't matter.
    m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    m_1stNullItemsBeginCount = 0;
    m_1stNullItemsMiddleCount = 0;
    m_2ndNullItemsCount = 0;
}

void VmaBlockMetadata_Linear::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
    VmaSuballocation& suballoc = FindSuballocation((VkDeviceSize)allocHandle - 1);
    suballoc.userData = userData;
}

void VmaBlockMetadata_Linear::DebugLogAllAllocations() const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    for (auto it = suballocations1st.begin() + m_1stNullItemsBeginCount; it != suballocations1st.end(); ++it)
        if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
            DebugLogAllocation(it->offset, it->size, it->userData);

    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();
    for (auto it = suballocations2nd.begin(); it != suballocations2nd.end(); ++it)
        if (it->type != VMA_SUBALLOCATION_TYPE_FREE)
            DebugLogAllocation(it->offset, it->size, it->userData);
}

VmaSuballocation& VmaBlockMetadata_Linear::FindSuballocation(VkDeviceSize offset) const
{
    const SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    const SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    VmaSuballocation refSuballoc;
    refSuballoc.offset = offset;
    // Rest of members stays uninitialized intentionally for better performance.

    // Item from the 1st vector.
    {
        SuballocationVectorType::const_iterator it = VmaBinaryFindSorted(
            suballocations1st.begin() + m_1stNullItemsBeginCount,
            suballocations1st.end(),
            refSuballoc,
            VmaSuballocationOffsetLess());
        if (it != suballocations1st.end())
        {
            return const_cast<VmaSuballocation&>(*it);
        }
    }

    if (m_2ndVectorMode != SECOND_VECTOR_EMPTY)
    {
        // Rest of members stays uninitialized intentionally for better performance.
        SuballocationVectorType::const_iterator it = m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER ?
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetLess()) :
            VmaBinaryFindSorted(suballocations2nd.begin(), suballocations2nd.end(), refSuballoc, VmaSuballocationOffsetGreater());
        if (it != suballocations2nd.end())
        {
            return const_cast<VmaSuballocation&>(*it);
        }
    }

    VMA_ASSERT(0 && "Allocation not found in linear allocator!");
    return const_cast<VmaSuballocation&>(suballocations1st.back()); // Should never occur.
}

bool VmaBlockMetadata_Linear::ShouldCompact1st() const
{
    const size_t nullItemCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
    const size_t suballocCount = AccessSuballocations1st().size();
    return suballocCount > 32 && nullItemCount * 2 >= (suballocCount - nullItemCount) * 3;
}

void VmaBlockMetadata_Linear::CleanupAfterFree()
{
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (IsEmpty())
    {
        suballocations1st.clear();
        suballocations2nd.clear();
        m_1stNullItemsBeginCount = 0;
        m_1stNullItemsMiddleCount = 0;
        m_2ndNullItemsCount = 0;
        m_2ndVectorMode = SECOND_VECTOR_EMPTY;
    }
    else
    {
        const size_t suballoc1stCount = suballocations1st.size();
        const size_t nullItem1stCount = m_1stNullItemsBeginCount + m_1stNullItemsMiddleCount;
        VMA_ASSERT(nullItem1stCount <= suballoc1stCount);

        // Find more null items at the beginning of 1st vector.
        while (m_1stNullItemsBeginCount < suballoc1stCount &&
            suballocations1st[m_1stNullItemsBeginCount].type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            ++m_1stNullItemsBeginCount;
            --m_1stNullItemsMiddleCount;
        }

        // Find more null items at the end of 1st vector.
        while (m_1stNullItemsMiddleCount > 0 &&
            suballocations1st.back().type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_1stNullItemsMiddleCount;
            suballocations1st.pop_back();
        }

        // Find more null items at the end of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd.back().type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            suballocations2nd.pop_back();
        }

        // Find more null items at the beginning of 2nd vector.
        while (m_2ndNullItemsCount > 0 &&
            suballocations2nd[0].type == VMA_SUBALLOCATION_TYPE_FREE)
        {
            --m_2ndNullItemsCount;
            VmaVectorRemove(suballocations2nd, 0);
        }

        if (ShouldCompact1st())
        {
            const size_t nonNullItemCount = suballoc1stCount - nullItem1stCount;
            size_t srcIndex = m_1stNullItemsBeginCount;
            for (size_t dstIndex = 0; dstIndex < nonNullItemCount; ++dstIndex)
            {
                while (suballocations1st[srcIndex].type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++srcIndex;
                }
                if (dstIndex != srcIndex)
                {
                    suballocations1st[dstIndex] = suballocations1st[srcIndex];
                }
                ++srcIndex;
            }
            suballocations1st.resize(nonNullItemCount);
            m_1stNullItemsBeginCount = 0;
            m_1stNullItemsMiddleCount = 0;
        }

        // 2nd vector became empty.
        if (suballocations2nd.empty())
        {
            m_2ndVectorMode = SECOND_VECTOR_EMPTY;
        }

        // 1st vector became empty.
        if (suballocations1st.size() - m_1stNullItemsBeginCount == 0)
        {
            suballocations1st.clear();
            m_1stNullItemsBeginCount = 0;

            if (!suballocations2nd.empty() && m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
            {
                // Swap 1st with 2nd. Now 2nd is empty.
                m_2ndVectorMode = SECOND_VECTOR_EMPTY;
                m_1stNullItemsMiddleCount = m_2ndNullItemsCount;
                while (m_1stNullItemsBeginCount < suballocations2nd.size() &&
                    suballocations2nd[m_1stNullItemsBeginCount].type == VMA_SUBALLOCATION_TYPE_FREE)
                {
                    ++m_1stNullItemsBeginCount;
                    --m_1stNullItemsMiddleCount;
                }
                m_2ndNullItemsCount = 0;
                m_1stVectorIndex ^= 1;
            }
        }
    }

    VMA_HEAVY_ASSERT(Validate());
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_LowerAddress(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = GetSize();
    const VkDeviceSize debugMargin = GetDebugMargin();
    const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
    {
        // Try to allocate at the end of 1st vector.

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations1st.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations1st.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations1st.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        const VkDeviceSize freeSpaceEnd = m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK ?
            suballocations2nd.back().offset : blockSize;

        // There is enough free space at the end after alignment.
        if (resultOffset + allocSize + debugMargin <= freeSpaceEnd)
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if ((allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity) && m_2ndVectorMode == SECOND_VECTOR_DOUBLE_STACK)
            {
                for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
                {
                    const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
                    if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on previous page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
            // pAllocationRequest->item, customData unused.
            pAllocationRequest->type = VmaAllocationRequestType::EndOf1st;
            return true;
        }
    }

    // Wrap-around to end of 2nd vector. Try to allocate there, watching for the
    // beginning of 1st vector as the end of free space.
    if (m_2ndVectorMode == SECOND_VECTOR_EMPTY || m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(!suballocations1st.empty());

        VkDeviceSize resultBaseOffset = 0;
        if (!suballocations2nd.empty())
        {
            const VmaSuballocation& lastSuballoc = suballocations2nd.back();
            resultBaseOffset = lastSuballoc.offset + lastSuballoc.size + debugMargin;
        }

        // Start from offset equal to beginning of free space.
        VkDeviceSize resultOffset = resultBaseOffset;

        // Apply alignment.
        resultOffset = VmaAlignUp(resultOffset, allocAlignment);

        // Check previous suballocations for BufferImageGranularity conflicts.
        // Make bigger alignment if necessary.
        if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
        {
            bool bufferImageGranularityConflict = false;
            for (size_t prevSuballocIndex = suballocations2nd.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations2nd[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(prevSuballoc.type, allocType))
                    {
                        bufferImageGranularityConflict = true;
                        break;
                    }
                }
                else
                    // Already on previous page.
                    break;
            }
            if (bufferImageGranularityConflict)
            {
                resultOffset = VmaAlignUp(resultOffset, bufferImageGranularity);
            }
        }

        size_t index1st = m_1stNullItemsBeginCount;

        // There is enough free space at the end after alignment.
        if ((index1st == suballocations1st.size() && resultOffset + allocSize + debugMargin <= blockSize) ||
            (index1st < suballocations1st.size() && resultOffset + allocSize + debugMargin <= suballocations1st[index1st].offset))
        {
            // Check next suballocations for BufferImageGranularity conflicts.
            // If conflict exists, allocation cannot be made here.
            if (allocSize % bufferImageGranularity || resultOffset % bufferImageGranularity)
            {
                for (size_t nextSuballocIndex = index1st;
                    nextSuballocIndex < suballocations1st.size();
                    nextSuballocIndex++)
                {
                    const VmaSuballocation& nextSuballoc = suballocations1st[nextSuballocIndex];
                    if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
                    {
                        if (VmaIsBufferImageGranularityConflict(allocType, nextSuballoc.type))
                        {
                            return false;
                        }
                    }
                    else
                    {
                        // Already on next page.
                        break;
                    }
                }
            }

            // All tests passed: Success.
            pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
            pAllocationRequest->type = VmaAllocationRequestType::EndOf2nd;
            // pAllocationRequest->item, customData unused.
            return true;
        }
    }

    return false;
}

bool VmaBlockMetadata_Linear::CreateAllocationRequest_UpperAddress(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    const VkDeviceSize blockSize = GetSize();
    const VkDeviceSize bufferImageGranularity = GetBufferImageGranularity();
    SuballocationVectorType& suballocations1st = AccessSuballocations1st();
    SuballocationVectorType& suballocations2nd = AccessSuballocations2nd();

    if (m_2ndVectorMode == SECOND_VECTOR_RING_BUFFER)
    {
        VMA_ASSERT(0 && "Trying to use pool with linear algorithm as double stack, while it is already being used as ring buffer.");
        return false;
    }

    // Try to allocate before 2nd.back(), or end of block if 2nd.empty().
    if (allocSize > blockSize)
    {
        return false;
    }
    VkDeviceSize resultBaseOffset = blockSize - allocSize;
    if (!suballocations2nd.empty())
    {
        const VmaSuballocation& lastSuballoc = suballocations2nd.back();
        resultBaseOffset = lastSuballoc.offset - allocSize;
        if (allocSize > lastSuballoc.offset)
        {
            return false;
        }
    }

    // Start from offset equal to end of free space.
    VkDeviceSize resultOffset = resultBaseOffset;

    const VkDeviceSize debugMargin = GetDebugMargin();

    // Apply debugMargin at the end.
    if (debugMargin > 0)
    {
        if (resultOffset < debugMargin)
        {
            return false;
        }
        resultOffset -= debugMargin;
    }

    // Apply alignment.
    resultOffset = VmaAlignDown(resultOffset, allocAlignment);

    // Check next suballocations from 2nd for BufferImageGranularity conflicts.
    // Make bigger alignment if necessary.
    if (bufferImageGranularity > 1 && bufferImageGranularity != allocAlignment && !suballocations2nd.empty())
    {
        bool bufferImageGranularityConflict = false;
        for (size_t nextSuballocIndex = suballocations2nd.size(); nextSuballocIndex--; )
        {
            const VmaSuballocation& nextSuballoc = suballocations2nd[nextSuballocIndex];
            if (VmaBlocksOnSamePage(resultOffset, allocSize, nextSuballoc.offset, bufferImageGranularity))
            {
                if (VmaIsBufferImageGranularityConflict(nextSuballoc.type, allocType))
                {
                    bufferImageGranularityConflict = true;
                    break;
                }
            }
            else
                // Already on previous page.
                break;
        }
        if (bufferImageGranularityConflict)
        {
            resultOffset = VmaAlignDown(resultOffset, bufferImageGranularity);
        }
    }

    // There is enough free space.
    const VkDeviceSize endOf1st = !suballocations1st.empty() ?
        suballocations1st.back().offset + suballocations1st.back().size :
        0;
    if (endOf1st + debugMargin <= resultOffset)
    {
        // Check previous suballocations for BufferImageGranularity conflicts.
        // If conflict exists, allocation cannot be made here.
        if (bufferImageGranularity > 1)
        {
            for (size_t prevSuballocIndex = suballocations1st.size(); prevSuballocIndex--; )
            {
                const VmaSuballocation& prevSuballoc = suballocations1st[prevSuballocIndex];
                if (VmaBlocksOnSamePage(prevSuballoc.offset, prevSuballoc.size, resultOffset, bufferImageGranularity))
                {
                    if (VmaIsBufferImageGranularityConflict(allocType, prevSuballoc.type))
                    {
                        return false;
                    }
                }
                else
                {
                    // Already on next page.
                    break;
                }
            }
        }

        // All tests passed: Success.
        pAllocationRequest->allocHandle = (VmaAllocHandle)(resultOffset + 1);
        // pAllocationRequest->item unused.
        pAllocationRequest->type = VmaAllocationRequestType::UpperAddress;
        return true;
    }

    return false;
}
#endif // _VMA_BLOCK_METADATA_LINEAR_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_LINEAR

#if 0
#ifndef _VMA_BLOCK_METADATA_BUDDY
/*
- GetSize() is the original size of allocated memory block.
- m_UsableSize is this size aligned down to a power of two.
  All allocations and calculations happen relative to m_UsableSize.
- GetUnusableSize() is the difference between them.
  It is reported as separate, unused range, not available for allocations.

Node at level 0 has size = m_UsableSize.
Each next level contains nodes with size 2 times smaller than current level.
m_LevelCount is the maximum number of levels to use in the current object.
*/
class VmaBlockMetadata_Buddy : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_Buddy)
public:
    VmaBlockMetadata_Buddy(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata_Buddy();

    size_t GetAllocationCount() const override { return m_AllocationCount; }
    VkDeviceSize GetSumFreeSize() const override { return m_SumFreeSize + GetUnusableSize(); }
    bool IsEmpty() const override { return m_Root->type == Node::TYPE_FREE; }
    VkResult CheckCorruption(const void* pBlockData) override { return VK_ERROR_FEATURE_NOT_PRESENT; }
    VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return (VkDeviceSize)allocHandle - 1; }
    void DebugLogAllAllocations() const override { DebugLogAllAllocationNode(m_Root, 0); }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;

    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json, uint32_t mapRefCount) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void Free(VmaAllocHandle allocHandle) override;
    void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle GetAllocationListBegin() const override;
    VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
    void Clear() override;
    void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;

private:
    static const size_t MAX_LEVELS = 48;

    struct ValidationContext
    {
        size_t calculatedAllocationCount = 0;
        size_t calculatedFreeCount = 0;
        VkDeviceSize calculatedSumFreeSize = 0;
    };
    struct Node
    {
        VkDeviceSize offset;
        enum TYPE
        {
            TYPE_FREE,
            TYPE_ALLOCATION,
            TYPE_SPLIT,
            TYPE_COUNT
        } type;
        Node* parent;
        Node* buddy;

        union
        {
            struct
            {
                Node* prev;
                Node* next;
            } free;
            struct
            {
                void* userData;
            } allocation;
            struct
            {
                Node* leftChild;
            } split;
        };
    };

    // Size of the memory block aligned down to a power of two.
    VkDeviceSize m_UsableSize;
    uint32_t m_LevelCount;
    VmaPoolAllocator<Node> m_NodeAllocator;
    Node* m_Root;
    struct
    {
        Node* front;
        Node* back;
    } m_FreeList[MAX_LEVELS];

    // Number of nodes in the tree with type == TYPE_ALLOCATION.
    size_t m_AllocationCount;
    // Number of nodes in the tree with type == TYPE_FREE.
    size_t m_FreeCount;
    // Doesn't include space wasted due to internal fragmentation - allocation sizes are just aligned up to node sizes.
    // Doesn't include unusable size.
    VkDeviceSize m_SumFreeSize;

    VkDeviceSize GetUnusableSize() const { return GetSize() - m_UsableSize; }
    VkDeviceSize LevelToNodeSize(uint32_t level) const { return m_UsableSize >> level; }

    VkDeviceSize AlignAllocationSize(VkDeviceSize size) const
    {
        if (!IsVirtual())
        {
            size = VmaAlignUp(size, (VkDeviceSize)16);
        }
        return VmaNextPow2(size);
    }
    Node* FindAllocationNode(VkDeviceSize offset, uint32_t& outLevel) const;
    void DeleteNodeChildren(Node* node);
    bool ValidateNode(ValidationContext& ctx, const Node* parent, const Node* curr, uint32_t level, VkDeviceSize levelNodeSize) const;
    uint32_t AllocSizeToLevel(VkDeviceSize allocSize) const;
    void AddNodeToDetailedStatistics(VmaDetailedStatistics& inoutStats, const Node* node, VkDeviceSize levelNodeSize) const;
    // Adds node to the front of FreeList at given level.
    // node->type must be FREE.
    // node->free.prev, next can be undefined.
    void AddToFreeListFront(uint32_t level, Node* node);
    // Removes node from FreeList at given level.
    // node->type must be FREE.
    // node->free.prev, next stay untouched.
    void RemoveFromFreeList(uint32_t level, Node* node);
    void DebugLogAllAllocationNode(Node* node, uint32_t level) const;

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMapNode(class VmaJsonWriter& json, const Node* node, VkDeviceSize levelNodeSize) const;
#endif
};

#ifndef _VMA_BLOCK_METADATA_BUDDY_FUNCTIONS
VmaBlockMetadata_Buddy::VmaBlockMetadata_Buddy(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_NodeAllocator(pAllocationCallbacks, 32), // firstBlockCapacity
    m_Root(VMA_NULL),
    m_AllocationCount(0),
    m_FreeCount(1),
    m_SumFreeSize(0)
{
    memset(m_FreeList, 0, sizeof(m_FreeList));
}

VmaBlockMetadata_Buddy::~VmaBlockMetadata_Buddy()
{
    DeleteNodeChildren(m_Root);
    m_NodeAllocator.Free(m_Root);
}

void VmaBlockMetadata_Buddy::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);

    m_UsableSize = VmaPrevPow2(size);
    m_SumFreeSize = m_UsableSize;

    // Calculate m_LevelCount.
    const VkDeviceSize minNodeSize = IsVirtual() ? 1 : 16;
    m_LevelCount = 1;
    while (m_LevelCount < MAX_LEVELS &&
        LevelToNodeSize(m_LevelCount) >= minNodeSize)
    {
        ++m_LevelCount;
    }

    Node* rootNode = m_NodeAllocator.Alloc();
    rootNode->offset = 0;
    rootNode->type = Node::TYPE_FREE;
    rootNode->parent = VMA_NULL;
    rootNode->buddy = VMA_NULL;

    m_Root = rootNode;
    AddToFreeListFront(0, rootNode);
}

bool VmaBlockMetadata_Buddy::Validate() const
{
    // Validate tree.
    ValidationContext ctx;
    if (!ValidateNode(ctx, VMA_NULL, m_Root, 0, LevelToNodeSize(0)))
    {
        VMA_VALIDATE(false && "ValidateNode failed.");
    }
    VMA_VALIDATE(m_AllocationCount == ctx.calculatedAllocationCount);
    VMA_VALIDATE(m_SumFreeSize == ctx.calculatedSumFreeSize);

    // Validate free node lists.
    for (uint32_t level = 0; level < m_LevelCount; ++level)
    {
        VMA_VALIDATE(m_FreeList[level].front == VMA_NULL ||
            m_FreeList[level].front->free.prev == VMA_NULL);

        for (Node* node = m_FreeList[level].front;
            node != VMA_NULL;
            node = node->free.next)
        {
            VMA_VALIDATE(node->type == Node::TYPE_FREE);

            if (node->free.next == VMA_NULL)
            {
                VMA_VALIDATE(m_FreeList[level].back == node);
            }
            else
            {
                VMA_VALIDATE(node->free.next->free.prev == node);
            }
        }
    }

    // Validate that free lists ar higher levels are empty.
    for (uint32_t level = m_LevelCount; level < MAX_LEVELS; ++level)
    {
        VMA_VALIDATE(m_FreeList[level].front == VMA_NULL && m_FreeList[level].back == VMA_NULL);
    }

    return true;
}

void VmaBlockMetadata_Buddy::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
{
    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += GetSize();

    AddNodeToDetailedStatistics(inoutStats, m_Root, LevelToNodeSize(0));

    const VkDeviceSize unusableSize = GetUnusableSize();
    if (unusableSize > 0)
        VmaAddDetailedStatisticsUnusedRange(inoutStats, unusableSize);
}

void VmaBlockMetadata_Buddy::AddStatistics(VmaStatistics& inoutStats) const
{
    inoutStats.blockCount++;
    inoutStats.allocationCount += (uint32_t)m_AllocationCount;
    inoutStats.blockBytes += GetSize();
    inoutStats.allocationBytes += GetSize() - m_SumFreeSize;
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Buddy::PrintDetailedMap(class VmaJsonWriter& json, uint32_t mapRefCount) const
{
    VmaDetailedStatistics stats;
    VmaClearDetailedStatistics(stats);
    AddDetailedStatistics(stats);

    PrintDetailedMap_Begin(
        json,
        stats.statistics.blockBytes - stats.statistics.allocationBytes,
        stats.statistics.allocationCount,
        stats.unusedRangeCount,
        mapRefCount);

    PrintDetailedMapNode(json, m_Root, LevelToNodeSize(0));

    const VkDeviceSize unusableSize = GetUnusableSize();
    if (unusableSize > 0)
    {
        PrintDetailedMap_UnusedRange(json,
            m_UsableSize, // offset
            unusableSize); // size
    }

    PrintDetailedMap_End(json);
}
#endif // VMA_STATS_STRING_ENABLED

bool VmaBlockMetadata_Buddy::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(!upperAddress && "VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

    allocSize = AlignAllocationSize(allocSize);

    // Simple way to respect bufferImageGranularity. May be optimized some day.
    // Whenever it might be an OPTIMAL image...
    if (allocType == VMA_SUBALLOCATION_TYPE_UNKNOWN ||
        allocType == VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN ||
        allocType == VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL)
    {
        allocAlignment = VMA_MAX(allocAlignment, GetBufferImageGranularity());
        allocSize = VmaAlignUp(allocSize, GetBufferImageGranularity());
    }

    if (allocSize > m_UsableSize)
    {
        return false;
    }

    const uint32_t targetLevel = AllocSizeToLevel(allocSize);
    for (uint32_t level = targetLevel; level--; )
    {
        for (Node* freeNode = m_FreeList[level].front;
            freeNode != VMA_NULL;
            freeNode = freeNode->free.next)
        {
            if (freeNode->offset % allocAlignment == 0)
            {
                pAllocationRequest->type = VmaAllocationRequestType::Normal;
                pAllocationRequest->allocHandle = (VmaAllocHandle)(freeNode->offset + 1);
                pAllocationRequest->size = allocSize;
                pAllocationRequest->customData = (void*)(uintptr_t)level;
                return true;
            }
        }
    }

    return false;
}

void VmaBlockMetadata_Buddy::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::Normal);

    const uint32_t targetLevel = AllocSizeToLevel(request.size);
    uint32_t currLevel = (uint32_t)(uintptr_t)request.customData;

    Node* currNode = m_FreeList[currLevel].front;
    VMA_ASSERT(currNode != VMA_NULL && currNode->type == Node::TYPE_FREE);
    const VkDeviceSize offset = (VkDeviceSize)request.allocHandle - 1;
    while (currNode->offset != offset)
    {
        currNode = currNode->free.next;
        VMA_ASSERT(currNode != VMA_NULL && currNode->type == Node::TYPE_FREE);
    }

    // Go down, splitting free nodes.
    while (currLevel < targetLevel)
    {
        // currNode is already first free node at currLevel.
        // Remove it from list of free nodes at this currLevel.
        RemoveFromFreeList(currLevel, currNode);

        const uint32_t childrenLevel = currLevel + 1;

        // Create two free sub-nodes.
        Node* leftChild = m_NodeAllocator.Alloc();
        Node* rightChild = m_NodeAllocator.Alloc();

        leftChild->offset = currNode->offset;
        leftChild->type = Node::TYPE_FREE;
        leftChild->parent = currNode;
        leftChild->buddy = rightChild;

        rightChild->offset = currNode->offset + LevelToNodeSize(childrenLevel);
        rightChild->type = Node::TYPE_FREE;
        rightChild->parent = currNode;
        rightChild->buddy = leftChild;

        // Convert current currNode to split type.
        currNode->type = Node::TYPE_SPLIT;
        currNode->split.leftChild = leftChild;

        // Add child nodes to free list. Order is important!
        AddToFreeListFront(childrenLevel, rightChild);
        AddToFreeListFront(childrenLevel, leftChild);

        ++m_FreeCount;
        ++currLevel;
        currNode = m_FreeList[currLevel].front;

        /*
        We can be sure that currNode, as left child of node previously split,
        also fulfills the alignment requirement.
        */
    }

    // Remove from free list.
    VMA_ASSERT(currLevel == targetLevel &&
        currNode != VMA_NULL &&
        currNode->type == Node::TYPE_FREE);
    RemoveFromFreeList(currLevel, currNode);

    // Convert to allocation node.
    currNode->type = Node::TYPE_ALLOCATION;
    currNode->allocation.userData = userData;

    ++m_AllocationCount;
    --m_FreeCount;
    m_SumFreeSize -= request.size;
}

void VmaBlockMetadata_Buddy::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    uint32_t level = 0;
    outInfo.offset = (VkDeviceSize)allocHandle - 1;
    const Node* const node = FindAllocationNode(outInfo.offset, level);
    outInfo.size = LevelToNodeSize(level);
    outInfo.pUserData = node->allocation.userData;
}

void* VmaBlockMetadata_Buddy::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
    uint32_t level = 0;
    const Node* const node = FindAllocationNode((VkDeviceSize)allocHandle - 1, level);
    return node->allocation.userData;
}

VmaAllocHandle VmaBlockMetadata_Buddy::GetAllocationListBegin() const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_Buddy::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
    // Function only used for defragmentation, which is disabled for this algorithm
    return VK_NULL_HANDLE;
}

void VmaBlockMetadata_Buddy::DeleteNodeChildren(Node* node)
{
    if (node->type == Node::TYPE_SPLIT)
    {
        DeleteNodeChildren(node->split.leftChild->buddy);
        DeleteNodeChildren(node->split.leftChild);
        const VkAllocationCallbacks* allocationCallbacks = GetAllocationCallbacks();
        m_NodeAllocator.Free(node->split.leftChild->buddy);
        m_NodeAllocator.Free(node->split.leftChild);
    }
}

void VmaBlockMetadata_Buddy::Clear()
{
    DeleteNodeChildren(m_Root);
    m_Root->type = Node::TYPE_FREE;
    m_AllocationCount = 0;
    m_FreeCount = 1;
    m_SumFreeSize = m_UsableSize;
}

void VmaBlockMetadata_Buddy::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
    uint32_t level = 0;
    Node* const node = FindAllocationNode((VkDeviceSize)allocHandle - 1, level);
    node->allocation.userData = userData;
}

VmaBlockMetadata_Buddy::Node* VmaBlockMetadata_Buddy::FindAllocationNode(VkDeviceSize offset, uint32_t& outLevel) const
{
    Node* node = m_Root;
    VkDeviceSize nodeOffset = 0;
    outLevel = 0;
    VkDeviceSize levelNodeSize = LevelToNodeSize(0);
    while (node->type == Node::TYPE_SPLIT)
    {
        const VkDeviceSize nextLevelNodeSize = levelNodeSize >> 1;
        if (offset < nodeOffset + nextLevelNodeSize)
        {
            node = node->split.leftChild;
        }
        else
        {
            node = node->split.leftChild->buddy;
            nodeOffset += nextLevelNodeSize;
        }
        ++outLevel;
        levelNodeSize = nextLevelNodeSize;
    }

    VMA_ASSERT(node != VMA_NULL && node->type == Node::TYPE_ALLOCATION);
    return node;
}

bool VmaBlockMetadata_Buddy::ValidateNode(ValidationContext& ctx, const Node* parent, const Node* curr, uint32_t level, VkDeviceSize levelNodeSize) const
{
    VMA_VALIDATE(level < m_LevelCount);
    VMA_VALIDATE(curr->parent == parent);
    VMA_VALIDATE((curr->buddy == VMA_NULL) == (parent == VMA_NULL));
    VMA_VALIDATE(curr->buddy == VMA_NULL || curr->buddy->buddy == curr);
    switch (curr->type)
    {
    case Node::TYPE_FREE:
        // curr->free.prev, next are validated separately.
        ctx.calculatedSumFreeSize += levelNodeSize;
        ++ctx.calculatedFreeCount;
        break;
    case Node::TYPE_ALLOCATION:
        ++ctx.calculatedAllocationCount;
        if (!IsVirtual())
        {
            VMA_VALIDATE(curr->allocation.userData != VMA_NULL);
        }
        break;
    case Node::TYPE_SPLIT:
    {
        const uint32_t childrenLevel = level + 1;
        const VkDeviceSize childrenLevelNodeSize = levelNodeSize >> 1;
        const Node* const leftChild = curr->split.leftChild;
        VMA_VALIDATE(leftChild != VMA_NULL);
        VMA_VALIDATE(leftChild->offset == curr->offset);
        if (!ValidateNode(ctx, curr, leftChild, childrenLevel, childrenLevelNodeSize))
        {
            VMA_VALIDATE(false && "ValidateNode for left child failed.");
        }
        const Node* const rightChild = leftChild->buddy;
        VMA_VALIDATE(rightChild->offset == curr->offset + childrenLevelNodeSize);
        if (!ValidateNode(ctx, curr, rightChild, childrenLevel, childrenLevelNodeSize))
        {
            VMA_VALIDATE(false && "ValidateNode for right child failed.");
        }
    }
    break;
    default:
        return false;
    }

    return true;
}

uint32_t VmaBlockMetadata_Buddy::AllocSizeToLevel(VkDeviceSize allocSize) const
{
    // I know this could be optimized somehow e.g. by using std::log2p1 from C++20.
    uint32_t level = 0;
    VkDeviceSize currLevelNodeSize = m_UsableSize;
    VkDeviceSize nextLevelNodeSize = currLevelNodeSize >> 1;
    while (allocSize <= nextLevelNodeSize && level + 1 < m_LevelCount)
    {
        ++level;
        currLevelNodeSize >>= 1;
        nextLevelNodeSize >>= 1;
    }
    return level;
}

void VmaBlockMetadata_Buddy::Free(VmaAllocHandle allocHandle)
{
    uint32_t level = 0;
    Node* node = FindAllocationNode((VkDeviceSize)allocHandle - 1, level);

    ++m_FreeCount;
    --m_AllocationCount;
    m_SumFreeSize += LevelToNodeSize(level);

    node->type = Node::TYPE_FREE;

    // Join free nodes if possible.
    while (level > 0 && node->buddy->type == Node::TYPE_FREE)
    {
        RemoveFromFreeList(level, node->buddy);
        Node* const parent = node->parent;

        m_NodeAllocator.Free(node->buddy);
        m_NodeAllocator.Free(node);
        parent->type = Node::TYPE_FREE;

        node = parent;
        --level;
        --m_FreeCount;
    }

    AddToFreeListFront(level, node);
}

void VmaBlockMetadata_Buddy::AddNodeToDetailedStatistics(VmaDetailedStatistics& inoutStats, const Node* node, VkDeviceSize levelNodeSize) const
{
    switch (node->type)
    {
    case Node::TYPE_FREE:
        VmaAddDetailedStatisticsUnusedRange(inoutStats, levelNodeSize);
        break;
    case Node::TYPE_ALLOCATION:
        VmaAddDetailedStatisticsAllocation(inoutStats, levelNodeSize);
        break;
    case Node::TYPE_SPLIT:
    {
        const VkDeviceSize childrenNodeSize = levelNodeSize / 2;
        const Node* const leftChild = node->split.leftChild;
        AddNodeToDetailedStatistics(inoutStats, leftChild, childrenNodeSize);
        const Node* const rightChild = leftChild->buddy;
        AddNodeToDetailedStatistics(inoutStats, rightChild, childrenNodeSize);
    }
    break;
    default:
        VMA_ASSERT(0);
    }
}

void VmaBlockMetadata_Buddy::AddToFreeListFront(uint32_t level, Node* node)
{
    VMA_ASSERT(node->type == Node::TYPE_FREE);

    // List is empty.
    Node* const frontNode = m_FreeList[level].front;
    if (frontNode == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].back == VMA_NULL);
        node->free.prev = node->free.next = VMA_NULL;
        m_FreeList[level].front = m_FreeList[level].back = node;
    }
    else
    {
        VMA_ASSERT(frontNode->free.prev == VMA_NULL);
        node->free.prev = VMA_NULL;
        node->free.next = frontNode;
        frontNode->free.prev = node;
        m_FreeList[level].front = node;
    }
}

void VmaBlockMetadata_Buddy::RemoveFromFreeList(uint32_t level, Node* node)
{
    VMA_ASSERT(m_FreeList[level].front != VMA_NULL);

    // It is at the front.
    if (node->free.prev == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].front == node);
        m_FreeList[level].front = node->free.next;
    }
    else
    {
        Node* const prevFreeNode = node->free.prev;
        VMA_ASSERT(prevFreeNode->free.next == node);
        prevFreeNode->free.next = node->free.next;
    }

    // It is at the back.
    if (node->free.next == VMA_NULL)
    {
        VMA_ASSERT(m_FreeList[level].back == node);
        m_FreeList[level].back = node->free.prev;
    }
    else
    {
        Node* const nextFreeNode = node->free.next;
        VMA_ASSERT(nextFreeNode->free.prev == node);
        nextFreeNode->free.prev = node->free.prev;
    }
}

void VmaBlockMetadata_Buddy::DebugLogAllAllocationNode(Node* node, uint32_t level) const
{
    switch (node->type)
    {
    case Node::TYPE_FREE:
        break;
    case Node::TYPE_ALLOCATION:
        DebugLogAllocation(node->offset, LevelToNodeSize(level), node->allocation.userData);
        break;
    case Node::TYPE_SPLIT:
    {
        ++level;
        DebugLogAllAllocationNode(node->split.leftChild, level);
        DebugLogAllAllocationNode(node->split.leftChild->buddy, level);
    }
    break;
    default:
        VMA_ASSERT(0);
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_Buddy::PrintDetailedMapNode(class VmaJsonWriter& json, const Node* node, VkDeviceSize levelNodeSize) const
{
    switch (node->type)
    {
    case Node::TYPE_FREE:
        PrintDetailedMap_UnusedRange(json, node->offset, levelNodeSize);
        break;
    case Node::TYPE_ALLOCATION:
        PrintDetailedMap_Allocation(json, node->offset, levelNodeSize, node->allocation.userData);
        break;
    case Node::TYPE_SPLIT:
    {
        const VkDeviceSize childrenNodeSize = levelNodeSize / 2;
        const Node* const leftChild = node->split.leftChild;
        PrintDetailedMapNode(json, leftChild, childrenNodeSize);
        const Node* const rightChild = leftChild->buddy;
        PrintDetailedMapNode(json, rightChild, childrenNodeSize);
    }
    break;
    default:
        VMA_ASSERT(0);
    }
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_BLOCK_METADATA_BUDDY_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_BUDDY
#endif // #if 0

#ifndef _VMA_BLOCK_METADATA_TLSF
// To not search current larger region if first allocation won't succeed and skip to smaller range
// use with VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT as strategy in CreateAllocationRequest().
// When fragmentation and reusal of previous blocks doesn't matter then use with
// VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT for fastest alloc time possible.
class VmaBlockMetadata_TLSF : public VmaBlockMetadata
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockMetadata_TLSF)
public:
    VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
        VkDeviceSize bufferImageGranularity, bool isVirtual);
    virtual ~VmaBlockMetadata_TLSF();

    size_t GetAllocationCount() const override { return m_AllocCount; }
    size_t GetFreeRegionsCount() const override { return m_BlocksFreeCount + 1; }
    VkDeviceSize GetSumFreeSize() const override { return m_BlocksFreeSize + m_NullBlock->size; }
    bool IsEmpty() const override { return m_NullBlock->offset == 0; }
    VkDeviceSize GetAllocationOffset(VmaAllocHandle allocHandle) const override { return ((Block*)allocHandle)->offset; }

    void Init(VkDeviceSize size) override;
    bool Validate() const override;

    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const override;
    void AddStatistics(VmaStatistics& inoutStats) const override;

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json) const override;
#endif

    bool CreateAllocationRequest(
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        bool upperAddress,
        VmaSuballocationType allocType,
        uint32_t strategy,
        VmaAllocationRequest* pAllocationRequest) override;

    VkResult CheckCorruption(const void* pBlockData) override;
    void Alloc(
        const VmaAllocationRequest& request,
        VmaSuballocationType type,
        void* userData) override;

    void Free(VmaAllocHandle allocHandle) override;
    void GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo) override;
    void* GetAllocationUserData(VmaAllocHandle allocHandle) const override;
    VmaAllocHandle GetAllocationListBegin() const override;
    VmaAllocHandle GetNextAllocation(VmaAllocHandle prevAlloc) const override;
    VkDeviceSize GetNextFreeRegionSize(VmaAllocHandle alloc) const override;
    void Clear() override;
    void SetAllocationUserData(VmaAllocHandle allocHandle, void* userData) override;
    void DebugLogAllAllocations() const override;

private:
    // According to original paper it should be preferable 4 or 5:
    // M. Masmano, I. Ripoll, A. Crespo, and J. Real "TLSF: a New Dynamic Memory Allocator for Real-Time Systems"
    // http://www.gii.upv.es/tlsf/files/ecrts04_tlsf.pdf
    static const uint8_t SECOND_LEVEL_INDEX = 5;
    static const uint16_t SMALL_BUFFER_SIZE = 256;
    static const uint32_t INITIAL_BLOCK_ALLOC_COUNT = 16;
    static const uint8_t MEMORY_CLASS_SHIFT = 7;
    static const uint8_t MAX_MEMORY_CLASSES = 65 - MEMORY_CLASS_SHIFT;

    class Block
    {
    public:
        VkDeviceSize offset;
        VkDeviceSize size;
        Block* prevPhysical;
        Block* nextPhysical;

        void MarkFree() { prevFree = VMA_NULL; }
        void MarkTaken() { prevFree = this; }
        bool IsFree() const { return prevFree != this; }
        void*& UserData() { VMA_HEAVY_ASSERT(!IsFree()); return userData; }
        Block*& PrevFree() { return prevFree; }
        Block*& NextFree() { VMA_HEAVY_ASSERT(IsFree()); return nextFree; }

    private:
        Block* prevFree; // Address of the same block here indicates that block is taken
        union
        {
            Block* nextFree;
            void* userData;
        };
    };

    size_t m_AllocCount;
    // Total number of free blocks besides null block
    size_t m_BlocksFreeCount;
    // Total size of free blocks excluding null block
    VkDeviceSize m_BlocksFreeSize;
    uint32_t m_IsFreeBitmap;
    uint8_t m_MemoryClasses;
    uint32_t m_InnerIsFreeBitmap[MAX_MEMORY_CLASSES];
    uint32_t m_ListsCount;
    /*
    * 0: 0-3 lists for small buffers
    * 1+: 0-(2^SLI-1) lists for normal buffers
    */
    Block** m_FreeList;
    VmaPoolAllocator<Block> m_BlockAllocator;
    Block* m_NullBlock;
    VmaBlockBufferImageGranularity m_GranularityHandler;

    uint8_t SizeToMemoryClass(VkDeviceSize size) const;
    uint16_t SizeToSecondIndex(VkDeviceSize size, uint8_t memoryClass) const;
    uint32_t GetListIndex(uint8_t memoryClass, uint16_t secondIndex) const;
    uint32_t GetListIndex(VkDeviceSize size) const;

    void RemoveFreeBlock(Block* block);
    void InsertFreeBlock(Block* block);
    void MergeBlock(Block* block, Block* prev);

    Block* FindFreeBlock(VkDeviceSize size, uint32_t& listIndex) const;
    bool CheckBlock(
        Block& block,
        uint32_t listIndex,
        VkDeviceSize allocSize,
        VkDeviceSize allocAlignment,
        VmaSuballocationType allocType,
        VmaAllocationRequest* pAllocationRequest);
};

#ifndef _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
VmaBlockMetadata_TLSF::VmaBlockMetadata_TLSF(const VkAllocationCallbacks* pAllocationCallbacks,
    VkDeviceSize bufferImageGranularity, bool isVirtual)
    : VmaBlockMetadata(pAllocationCallbacks, bufferImageGranularity, isVirtual),
    m_AllocCount(0),
    m_BlocksFreeCount(0),
    m_BlocksFreeSize(0),
    m_IsFreeBitmap(0),
    m_MemoryClasses(0),
    m_ListsCount(0),
    m_FreeList(VMA_NULL),
    m_BlockAllocator(pAllocationCallbacks, INITIAL_BLOCK_ALLOC_COUNT),
    m_NullBlock(VMA_NULL),
    m_GranularityHandler(bufferImageGranularity) {}

VmaBlockMetadata_TLSF::~VmaBlockMetadata_TLSF()
{
    if (m_FreeList)
        vma_delete_array(GetAllocationCallbacks(), m_FreeList, m_ListsCount);
    m_GranularityHandler.Destroy(GetAllocationCallbacks());
}

void VmaBlockMetadata_TLSF::Init(VkDeviceSize size)
{
    VmaBlockMetadata::Init(size);

    if (!IsVirtual())
        m_GranularityHandler.Init(GetAllocationCallbacks(), size);

    m_NullBlock = m_BlockAllocator.Alloc();
    m_NullBlock->size = size;
    m_NullBlock->offset = 0;
    m_NullBlock->prevPhysical = VMA_NULL;
    m_NullBlock->nextPhysical = VMA_NULL;
    m_NullBlock->MarkFree();
    m_NullBlock->NextFree() = VMA_NULL;
    m_NullBlock->PrevFree() = VMA_NULL;
    uint8_t memoryClass = SizeToMemoryClass(size);
    uint16_t sli = SizeToSecondIndex(size, memoryClass);
    m_ListsCount = (memoryClass == 0 ? 0 : (memoryClass - 1) * (1UL << SECOND_LEVEL_INDEX) + sli) + 1;
    if (IsVirtual())
        m_ListsCount += 1UL << SECOND_LEVEL_INDEX;
    else
        m_ListsCount += 4;

    m_MemoryClasses = memoryClass + uint8_t(2);
    memset(m_InnerIsFreeBitmap, 0, MAX_MEMORY_CLASSES * sizeof(uint32_t));

    m_FreeList = vma_new_array(GetAllocationCallbacks(), Block*, m_ListsCount);
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
}

bool VmaBlockMetadata_TLSF::Validate() const
{
    VMA_VALIDATE(GetSumFreeSize() <= GetSize());

    VkDeviceSize calculatedSize = m_NullBlock->size;
    VkDeviceSize calculatedFreeSize = m_NullBlock->size;
    size_t allocCount = 0;
    size_t freeCount = 0;

    // Check integrity of free lists
    for (uint32_t list = 0; list < m_ListsCount; ++list)
    {
        Block* block = m_FreeList[list];
        if (block != VMA_NULL)
        {
            VMA_VALIDATE(block->IsFree());
            VMA_VALIDATE(block->PrevFree() == VMA_NULL);
            while (block->NextFree())
            {
                VMA_VALIDATE(block->NextFree()->IsFree());
                VMA_VALIDATE(block->NextFree()->PrevFree() == block);
                block = block->NextFree();
            }
        }
    }

    VkDeviceSize nextOffset = m_NullBlock->offset;
    auto validateCtx = m_GranularityHandler.StartValidation(GetAllocationCallbacks(), IsVirtual());

    VMA_VALIDATE(m_NullBlock->nextPhysical == VMA_NULL);
    if (m_NullBlock->prevPhysical)
    {
        VMA_VALIDATE(m_NullBlock->prevPhysical->nextPhysical == m_NullBlock);
    }
    // Check all blocks
    for (Block* prev = m_NullBlock->prevPhysical; prev != VMA_NULL; prev = prev->prevPhysical)
    {
        VMA_VALIDATE(prev->offset + prev->size == nextOffset);
        nextOffset = prev->offset;
        calculatedSize += prev->size;

        uint32_t listIndex = GetListIndex(prev->size);
        if (prev->IsFree())
        {
            ++freeCount;
            // Check if free block belongs to free list
            Block* freeBlock = m_FreeList[listIndex];
            VMA_VALIDATE(freeBlock != VMA_NULL);

            bool found = false;
            do
            {
                if (freeBlock == prev)
                    found = true;

                freeBlock = freeBlock->NextFree();
            } while (!found && freeBlock != VMA_NULL);

            VMA_VALIDATE(found);
            calculatedFreeSize += prev->size;
        }
        else
        {
            ++allocCount;
            // Check if taken block is not on a free list
            Block* freeBlock = m_FreeList[listIndex];
            while (freeBlock)
            {
                VMA_VALIDATE(freeBlock != prev);
                freeBlock = freeBlock->NextFree();
            }

            if (!IsVirtual())
            {
                VMA_VALIDATE(m_GranularityHandler.Validate(validateCtx, prev->offset, prev->size));
            }
        }

        if (prev->prevPhysical)
        {
            VMA_VALIDATE(prev->prevPhysical->nextPhysical == prev);
        }
    }

    if (!IsVirtual())
    {
        VMA_VALIDATE(m_GranularityHandler.FinishValidation(validateCtx));
    }

    VMA_VALIDATE(nextOffset == 0);
    VMA_VALIDATE(calculatedSize == GetSize());
    VMA_VALIDATE(calculatedFreeSize == GetSumFreeSize());
    VMA_VALIDATE(allocCount == m_AllocCount);
    VMA_VALIDATE(freeCount == m_BlocksFreeCount);

    return true;
}

void VmaBlockMetadata_TLSF::AddDetailedStatistics(VmaDetailedStatistics& inoutStats) const
{
    inoutStats.statistics.blockCount++;
    inoutStats.statistics.blockBytes += GetSize();
    if (m_NullBlock->size > 0)
        VmaAddDetailedStatisticsUnusedRange(inoutStats, m_NullBlock->size);

    for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        if (block->IsFree())
            VmaAddDetailedStatisticsUnusedRange(inoutStats, block->size);
        else
            VmaAddDetailedStatisticsAllocation(inoutStats, block->size);
    }
}

void VmaBlockMetadata_TLSF::AddStatistics(VmaStatistics& inoutStats) const
{
    inoutStats.blockCount++;
    inoutStats.allocationCount += (uint32_t)m_AllocCount;
    inoutStats.blockBytes += GetSize();
    inoutStats.allocationBytes += GetSize() - GetSumFreeSize();
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockMetadata_TLSF::PrintDetailedMap(class VmaJsonWriter& json) const
{
    size_t blockCount = m_AllocCount + m_BlocksFreeCount;
    VmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
    VmaVector<Block*, VmaStlAllocator<Block*>> blockList(blockCount, allocator);

    size_t i = blockCount;
    for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        blockList[--i] = block;
    }
    VMA_ASSERT(i == 0);

    VmaDetailedStatistics stats;
    VmaClearDetailedStatistics(stats);
    AddDetailedStatistics(stats);

    PrintDetailedMap_Begin(json,
        stats.statistics.blockBytes - stats.statistics.allocationBytes,
        stats.statistics.allocationCount,
        stats.unusedRangeCount);

    for (; i < blockCount; ++i)
    {
        Block* block = blockList[i];
        if (block->IsFree())
            PrintDetailedMap_UnusedRange(json, block->offset, block->size);
        else
            PrintDetailedMap_Allocation(json, block->offset, block->size, block->UserData());
    }
    if (m_NullBlock->size > 0)
        PrintDetailedMap_UnusedRange(json, m_NullBlock->offset, m_NullBlock->size);

    PrintDetailedMap_End(json);
}
#endif

bool VmaBlockMetadata_TLSF::CreateAllocationRequest(
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    bool upperAddress,
    VmaSuballocationType allocType,
    uint32_t strategy,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(allocSize > 0 && "Cannot allocate empty block!");
    VMA_ASSERT(!upperAddress && "VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT can be used only with linear algorithm.");

    // For small granularity round up
    if (!IsVirtual())
        m_GranularityHandler.RoundupAllocRequest(allocType, allocSize, allocAlignment);

    allocSize += GetDebugMargin();
    // Quick check for too small pool
    if (allocSize > GetSumFreeSize())
        return false;

    // If no free blocks in pool then check only null block
    if (m_BlocksFreeCount == 0)
        return CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest);

    // Round up to the next block
    VkDeviceSize sizeForNextList = allocSize;
    VkDeviceSize smallSizeStep = VkDeviceSize(SMALL_BUFFER_SIZE / (IsVirtual() ? 1 << SECOND_LEVEL_INDEX : 4));
    if (allocSize > SMALL_BUFFER_SIZE)
    {
        sizeForNextList += (1ULL << (VMA_BITSCAN_MSB(allocSize) - SECOND_LEVEL_INDEX));
    }
    else if (allocSize > SMALL_BUFFER_SIZE - smallSizeStep)
        sizeForNextList = SMALL_BUFFER_SIZE + 1;
    else
        sizeForNextList += smallSizeStep;

    uint32_t nextListIndex = 0;
    uint32_t prevListIndex = 0;
    Block* nextListBlock = VMA_NULL;
    Block* prevListBlock = VMA_NULL;

    // Check blocks according to strategies
    if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT)
    {
        // Quick check for larger block first
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        if (nextListBlock != VMA_NULL && CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // If not fitted then null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Null block failed, search larger bucket
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }

        // Failed again, check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }
    }
    else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT)
    {
        // Check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check larger bucket
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }
    }
    else if (strategy & VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT )
    {
        // Perform search from the start
        VmaStlAllocator<Block*> allocator(GetAllocationCallbacks());
        VmaVector<Block*, VmaStlAllocator<Block*>> blockList(m_BlocksFreeCount, allocator);

        size_t i = m_BlocksFreeCount;
        for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
        {
            if (block->IsFree() && block->size >= allocSize)
                blockList[--i] = block;
        }

        for (; i < m_BlocksFreeCount; ++i)
        {
            Block& block = *blockList[i];
            if (CheckBlock(block, GetListIndex(block.size), allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Whole range searched, no more memory
        return false;
    }
    else
    {
        // Check larger bucket
        nextListBlock = FindFreeBlock(sizeForNextList, nextListIndex);
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }

        // If failed check null block
        if (CheckBlock(*m_NullBlock, m_ListsCount, allocSize, allocAlignment, allocType, pAllocationRequest))
            return true;

        // Check best fit bucket
        prevListBlock = FindFreeBlock(allocSize, prevListIndex);
        while (prevListBlock)
        {
            if (CheckBlock(*prevListBlock, prevListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            prevListBlock = prevListBlock->NextFree();
        }
    }

    // Worst case, full search has to be done
    while (++nextListIndex < m_ListsCount)
    {
        nextListBlock = m_FreeList[nextListIndex];
        while (nextListBlock)
        {
            if (CheckBlock(*nextListBlock, nextListIndex, allocSize, allocAlignment, allocType, pAllocationRequest))
                return true;
            nextListBlock = nextListBlock->NextFree();
        }
    }

    // No more memory sadly
    return false;
}

VkResult VmaBlockMetadata_TLSF::CheckCorruption(const void* pBlockData)
{
    for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
    {
        if (!block->IsFree())
        {
            if (!VmaValidateMagicValue(pBlockData, block->offset + block->size))
            {
                VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER VALIDATED ALLOCATION!");
                return VK_ERROR_UNKNOWN_COPY;
            }
        }
    }

    return VK_SUCCESS;
}

void VmaBlockMetadata_TLSF::Alloc(
    const VmaAllocationRequest& request,
    VmaSuballocationType type,
    void* userData)
{
    VMA_ASSERT(request.type == VmaAllocationRequestType::TLSF);

    // Get block and pop it from the free list
    Block* currentBlock = (Block*)request.allocHandle;
    VkDeviceSize offset = request.algorithmData;
    VMA_ASSERT(currentBlock != VMA_NULL);
    VMA_ASSERT(currentBlock->offset <= offset);

    if (currentBlock != m_NullBlock)
        RemoveFreeBlock(currentBlock);

    VkDeviceSize debugMargin = GetDebugMargin();
    VkDeviceSize misssingAlignment = offset - currentBlock->offset;

    // Append missing alignment to prev block or create new one
    if (misssingAlignment)
    {
        Block* prevBlock = currentBlock->prevPhysical;
        VMA_ASSERT(prevBlock != VMA_NULL && "There should be no missing alignment at offset 0!");

        if (prevBlock->IsFree() && prevBlock->size != debugMargin)
        {
            uint32_t oldList = GetListIndex(prevBlock->size);
            prevBlock->size += misssingAlignment;
            // Check if new size crosses list bucket
            if (oldList != GetListIndex(prevBlock->size))
            {
                prevBlock->size -= misssingAlignment;
                RemoveFreeBlock(prevBlock);
                prevBlock->size += misssingAlignment;
                InsertFreeBlock(prevBlock);
            }
            else
                m_BlocksFreeSize += misssingAlignment;
        }
        else
        {
            Block* newBlock = m_BlockAllocator.Alloc();
            currentBlock->prevPhysical = newBlock;
            prevBlock->nextPhysical = newBlock;
            newBlock->prevPhysical = prevBlock;
            newBlock->nextPhysical = currentBlock;
            newBlock->size = misssingAlignment;
            newBlock->offset = currentBlock->offset;
            newBlock->MarkTaken();

            InsertFreeBlock(newBlock);
        }

        currentBlock->size -= misssingAlignment;
        currentBlock->offset += misssingAlignment;
    }

    VkDeviceSize size = request.size + debugMargin;
    if (currentBlock->size == size)
    {
        if (currentBlock == m_NullBlock)
        {
            // Setup new null block
            m_NullBlock = m_BlockAllocator.Alloc();
            m_NullBlock->size = 0;
            m_NullBlock->offset = currentBlock->offset + size;
            m_NullBlock->prevPhysical = currentBlock;
            m_NullBlock->nextPhysical = VMA_NULL;
            m_NullBlock->MarkFree();
            m_NullBlock->PrevFree() = VMA_NULL;
            m_NullBlock->NextFree() = VMA_NULL;
            currentBlock->nextPhysical = m_NullBlock;
            currentBlock->MarkTaken();
        }
    }
    else
    {
        VMA_ASSERT(currentBlock->size > size && "Proper block already found, shouldn't find smaller one!");

        // Create new free block
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = currentBlock->size - size;
        newBlock->offset = currentBlock->offset + size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        currentBlock->nextPhysical = newBlock;
        currentBlock->size = size;

        if (currentBlock == m_NullBlock)
        {
            m_NullBlock = newBlock;
            m_NullBlock->MarkFree();
            m_NullBlock->NextFree() = VMA_NULL;
            m_NullBlock->PrevFree() = VMA_NULL;
            currentBlock->MarkTaken();
        }
        else
        {
            newBlock->nextPhysical->prevPhysical = newBlock;
            newBlock->MarkTaken();
            InsertFreeBlock(newBlock);
        }
    }
    currentBlock->UserData() = userData;

    if (debugMargin > 0)
    {
        currentBlock->size -= debugMargin;
        Block* newBlock = m_BlockAllocator.Alloc();
        newBlock->size = debugMargin;
        newBlock->offset = currentBlock->offset + currentBlock->size;
        newBlock->prevPhysical = currentBlock;
        newBlock->nextPhysical = currentBlock->nextPhysical;
        newBlock->MarkTaken();
        currentBlock->nextPhysical->prevPhysical = newBlock;
        currentBlock->nextPhysical = newBlock;
        InsertFreeBlock(newBlock);
    }

    if (!IsVirtual())
        m_GranularityHandler.AllocPages((uint8_t)(uintptr_t)request.customData,
            currentBlock->offset, currentBlock->size);
    ++m_AllocCount;
}

void VmaBlockMetadata_TLSF::Free(VmaAllocHandle allocHandle)
{
    Block* block = (Block*)allocHandle;
    Block* next = block->nextPhysical;
    VMA_ASSERT(!block->IsFree() && "Block is already free!");

    if (!IsVirtual())
        m_GranularityHandler.FreePages(block->offset, block->size);
    --m_AllocCount;

    VkDeviceSize debugMargin = GetDebugMargin();
    if (debugMargin > 0)
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        block = next;
        next = next->nextPhysical;
    }

    // Try merging
    Block* prev = block->prevPhysical;
    if (prev != VMA_NULL && prev->IsFree() && prev->size != debugMargin)
    {
        RemoveFreeBlock(prev);
        MergeBlock(block, prev);
    }

    if (!next->IsFree())
        InsertFreeBlock(block);
    else if (next == m_NullBlock)
        MergeBlock(m_NullBlock, block);
    else
    {
        RemoveFreeBlock(next);
        MergeBlock(next, block);
        InsertFreeBlock(next);
    }
}

void VmaBlockMetadata_TLSF::GetAllocationInfo(VmaAllocHandle allocHandle, VmaVirtualAllocationInfo& outInfo)
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->IsFree() && "Cannot get allocation info for free block!");
    outInfo.offset = block->offset;
    outInfo.size = block->size;
    outInfo.pUserData = block->UserData();
}

void* VmaBlockMetadata_TLSF::GetAllocationUserData(VmaAllocHandle allocHandle) const
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->IsFree() && "Cannot get user data for free block!");
    return block->UserData();
}

VmaAllocHandle VmaBlockMetadata_TLSF::GetAllocationListBegin() const
{
    if (m_AllocCount == 0)
        return VK_NULL_HANDLE;

    for (Block* block = m_NullBlock->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->IsFree())
            return (VmaAllocHandle)block;
    }
    VMA_ASSERT(false && "If m_AllocCount > 0 then should find any allocation!");
    return VK_NULL_HANDLE;
}

VmaAllocHandle VmaBlockMetadata_TLSF::GetNextAllocation(VmaAllocHandle prevAlloc) const
{
    Block* startBlock = (Block*)prevAlloc;
    VMA_ASSERT(!startBlock->IsFree() && "Incorrect block!");

    for (Block* block = startBlock->prevPhysical; block; block = block->prevPhysical)
    {
        if (!block->IsFree())
            return (VmaAllocHandle)block;
    }
    return VK_NULL_HANDLE;
}

VkDeviceSize VmaBlockMetadata_TLSF::GetNextFreeRegionSize(VmaAllocHandle alloc) const
{
    Block* block = (Block*)alloc;
    VMA_ASSERT(!block->IsFree() && "Incorrect block!");

    if (block->prevPhysical)
        return block->prevPhysical->IsFree() ? block->prevPhysical->size : 0;
    return 0;
}

void VmaBlockMetadata_TLSF::Clear()
{
    m_AllocCount = 0;
    m_BlocksFreeCount = 0;
    m_BlocksFreeSize = 0;
    m_IsFreeBitmap = 0;
    m_NullBlock->offset = 0;
    m_NullBlock->size = GetSize();
    Block* block = m_NullBlock->prevPhysical;
    m_NullBlock->prevPhysical = VMA_NULL;
    while (block)
    {
        Block* prev = block->prevPhysical;
        m_BlockAllocator.Free(block);
        block = prev;
    }
    memset(m_FreeList, 0, m_ListsCount * sizeof(Block*));
    memset(m_InnerIsFreeBitmap, 0, m_MemoryClasses * sizeof(uint32_t));
    m_GranularityHandler.Clear();
}

void VmaBlockMetadata_TLSF::SetAllocationUserData(VmaAllocHandle allocHandle, void* userData)
{
    Block* block = (Block*)allocHandle;
    VMA_ASSERT(!block->IsFree() && "Trying to set user data for not allocated block!");
    block->UserData() = userData;
}

void VmaBlockMetadata_TLSF::DebugLogAllAllocations() const
{
    for (Block* block = m_NullBlock->prevPhysical; block != VMA_NULL; block = block->prevPhysical)
        if (!block->IsFree())
            DebugLogAllocation(block->offset, block->size, block->UserData());
}

uint8_t VmaBlockMetadata_TLSF::SizeToMemoryClass(VkDeviceSize size) const
{
    if (size > SMALL_BUFFER_SIZE)
        return uint8_t(VMA_BITSCAN_MSB(size) - MEMORY_CLASS_SHIFT);
    return 0;
}

uint16_t VmaBlockMetadata_TLSF::SizeToSecondIndex(VkDeviceSize size, uint8_t memoryClass) const
{
    if (memoryClass == 0)
    {
        if (IsVirtual())
            return static_cast<uint16_t>((size - 1) / 8);
        else
            return static_cast<uint16_t>((size - 1) / 64);
    }
    return static_cast<uint16_t>((size >> (memoryClass + MEMORY_CLASS_SHIFT - SECOND_LEVEL_INDEX)) ^ (1U << SECOND_LEVEL_INDEX));
}

uint32_t VmaBlockMetadata_TLSF::GetListIndex(uint8_t memoryClass, uint16_t secondIndex) const
{
    if (memoryClass == 0)
        return secondIndex;

    const uint32_t index = static_cast<uint32_t>(memoryClass - 1) * (1 << SECOND_LEVEL_INDEX) + secondIndex;
    if (IsVirtual())
        return index + (1 << SECOND_LEVEL_INDEX);
    else
        return index + 4;
}

uint32_t VmaBlockMetadata_TLSF::GetListIndex(VkDeviceSize size) const
{
    uint8_t memoryClass = SizeToMemoryClass(size);
    return GetListIndex(memoryClass, SizeToSecondIndex(size, memoryClass));
}

void VmaBlockMetadata_TLSF::RemoveFreeBlock(Block* block)
{
    VMA_ASSERT(block != m_NullBlock);
    VMA_ASSERT(block->IsFree());

    if (block->NextFree() != VMA_NULL)
        block->NextFree()->PrevFree() = block->PrevFree();
    if (block->PrevFree() != VMA_NULL)
        block->PrevFree()->NextFree() = block->NextFree();
    else
    {
        uint8_t memClass = SizeToMemoryClass(block->size);
        uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
        uint32_t index = GetListIndex(memClass, secondIndex);
        VMA_ASSERT(m_FreeList[index] == block);
        m_FreeList[index] = block->NextFree();
        if (block->NextFree() == VMA_NULL)
        {
            m_InnerIsFreeBitmap[memClass] &= ~(1U << secondIndex);
            if (m_InnerIsFreeBitmap[memClass] == 0)
                m_IsFreeBitmap &= ~(1UL << memClass);
        }
    }
    block->MarkTaken();
    block->UserData() = VMA_NULL;
    --m_BlocksFreeCount;
    m_BlocksFreeSize -= block->size;
}

void VmaBlockMetadata_TLSF::InsertFreeBlock(Block* block)
{
    VMA_ASSERT(block != m_NullBlock);
    VMA_ASSERT(!block->IsFree() && "Cannot insert block twice!");

    uint8_t memClass = SizeToMemoryClass(block->size);
    uint16_t secondIndex = SizeToSecondIndex(block->size, memClass);
    uint32_t index = GetListIndex(memClass, secondIndex);
    VMA_ASSERT(index < m_ListsCount);
    block->PrevFree() = VMA_NULL;
    block->NextFree() = m_FreeList[index];
    m_FreeList[index] = block;
    if (block->NextFree() != VMA_NULL)
        block->NextFree()->PrevFree() = block;
    else
    {
        m_InnerIsFreeBitmap[memClass] |= 1U << secondIndex;
        m_IsFreeBitmap |= 1UL << memClass;
    }
    ++m_BlocksFreeCount;
    m_BlocksFreeSize += block->size;
}

void VmaBlockMetadata_TLSF::MergeBlock(Block* block, Block* prev)
{
    VMA_ASSERT(block->prevPhysical == prev && "Cannot merge separate physical regions!");
    VMA_ASSERT(!prev->IsFree() && "Cannot merge block that belongs to free list!");

    block->offset = prev->offset;
    block->size += prev->size;
    block->prevPhysical = prev->prevPhysical;
    if (block->prevPhysical)
        block->prevPhysical->nextPhysical = block;
    m_BlockAllocator.Free(prev);
}

VmaBlockMetadata_TLSF::Block* VmaBlockMetadata_TLSF::FindFreeBlock(VkDeviceSize size, uint32_t& listIndex) const
{
    uint8_t memoryClass = SizeToMemoryClass(size);
    uint32_t innerFreeMap = m_InnerIsFreeBitmap[memoryClass] & (~0U << SizeToSecondIndex(size, memoryClass));
    if (!innerFreeMap)
    {
        // Check higher levels for available blocks
        uint32_t freeMap = m_IsFreeBitmap & (~0UL << (memoryClass + 1));
        if (!freeMap)
            return VMA_NULL; // No more memory available

        // Find lowest free region
        memoryClass = VMA_BITSCAN_LSB(freeMap);
        innerFreeMap = m_InnerIsFreeBitmap[memoryClass];
        VMA_ASSERT(innerFreeMap != 0);
    }
    // Find lowest free subregion
    listIndex = GetListIndex(memoryClass, VMA_BITSCAN_LSB(innerFreeMap));
    VMA_ASSERT(m_FreeList[listIndex]);
    return m_FreeList[listIndex];
}

bool VmaBlockMetadata_TLSF::CheckBlock(
    Block& block,
    uint32_t listIndex,
    VkDeviceSize allocSize,
    VkDeviceSize allocAlignment,
    VmaSuballocationType allocType,
    VmaAllocationRequest* pAllocationRequest)
{
    VMA_ASSERT(block.IsFree() && "Block is already taken!");

    VkDeviceSize alignedOffset = VmaAlignUp(block.offset, allocAlignment);
    if (block.size < allocSize + alignedOffset - block.offset)
        return false;

    // Check for granularity conflicts
    if (!IsVirtual() &&
        m_GranularityHandler.CheckConflictAndAlignUp(alignedOffset, allocSize, block.offset, block.size, allocType))
        return false;

    // Alloc successful
    pAllocationRequest->type = VmaAllocationRequestType::TLSF;
    pAllocationRequest->allocHandle = (VmaAllocHandle)&block;
    pAllocationRequest->size = allocSize - GetDebugMargin();
    pAllocationRequest->customData = (void*)allocType;
    pAllocationRequest->algorithmData = alignedOffset;

    // Place block at the start of list if it's normal block
    if (listIndex != m_ListsCount && block.PrevFree())
    {
        block.PrevFree()->NextFree() = block.NextFree();
        if (block.NextFree())
            block.NextFree()->PrevFree() = block.PrevFree();
        block.PrevFree() = VMA_NULL;
        block.NextFree() = m_FreeList[listIndex];
        m_FreeList[listIndex] = &block;
        if (block.NextFree())
            block.NextFree()->PrevFree() = &block;
    }

    return true;
}
#endif // _VMA_BLOCK_METADATA_TLSF_FUNCTIONS
#endif // _VMA_BLOCK_METADATA_TLSF

#ifndef _VMA_BLOCK_VECTOR
/*
Sequence of VmaDeviceMemoryBlock. Represents memory blocks allocated for a specific
Vulkan memory type.

Synchronized internally with a mutex.
*/
class VmaBlockVector
{
    friend struct VmaDefragmentationContext_T;
    VMA_CLASS_NO_COPY_NO_MOVE(VmaBlockVector)
public:
    VmaBlockVector(
        VmaAllocator hAllocator,
        VmaPool hParentPool,
        uint32_t memoryTypeIndex,
        VkDeviceSize preferredBlockSize,
        size_t minBlockCount,
        size_t maxBlockCount,
        VkDeviceSize bufferImageGranularity,
        bool explicitBlockSize,
        uint32_t algorithm,
        float priority,
        VkDeviceSize minAllocationAlignment,
        void* pMemoryAllocateNext);
    ~VmaBlockVector();

    VmaAllocator GetAllocator() const { return m_hAllocator; }
    VmaPool GetParentPool() const { return m_hParentPool; }
    bool IsCustomPool() const { return m_hParentPool != VMA_NULL; }
    uint32_t GetMemoryTypeIndex() const { return m_MemoryTypeIndex; }
    VkDeviceSize GetPreferredBlockSize() const { return m_PreferredBlockSize; }
    VkDeviceSize GetBufferImageGranularity() const { return m_BufferImageGranularity; }
    uint32_t GetAlgorithm() const { return m_Algorithm; }
    bool HasExplicitBlockSize() const { return m_ExplicitBlockSize; }
    float GetPriority() const { return m_Priority; }
    const void* GetAllocationNextPtr() const { return m_pMemoryAllocateNext; }
    // To be used only while the m_Mutex is locked. Used during defragmentation.
    size_t GetBlockCount() const { return m_Blocks.size(); }
    // To be used only while the m_Mutex is locked. Used during defragmentation.
    VmaDeviceMemoryBlock* GetBlock(size_t index) const { return m_Blocks[index]; }
    VMA_RW_MUTEX &GetMutex() { return m_Mutex; }

    VkResult CreateMinBlocks();
    void AddStatistics(VmaStatistics& inoutStats);
    void AddDetailedStatistics(VmaDetailedStatistics& inoutStats);
    bool IsEmpty();
    bool IsCorruptionDetectionEnabled() const;

    VkResult Allocate(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    void Free(const VmaAllocation hAllocation);

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json);
#endif

    VkResult CheckCorruption();

private:
    const VmaAllocator m_hAllocator;
    const VmaPool m_hParentPool;
    const uint32_t m_MemoryTypeIndex;
    const VkDeviceSize m_PreferredBlockSize;
    const size_t m_MinBlockCount;
    const size_t m_MaxBlockCount;
    const VkDeviceSize m_BufferImageGranularity;
    const bool m_ExplicitBlockSize;
    const uint32_t m_Algorithm;
    const float m_Priority;
    const VkDeviceSize m_MinAllocationAlignment;

    void* const m_pMemoryAllocateNext;
    VMA_RW_MUTEX m_Mutex;
    // Incrementally sorted by sumFreeSize, ascending.
    VmaVector<VmaDeviceMemoryBlock*, VmaStlAllocator<VmaDeviceMemoryBlock*>> m_Blocks;
    uint32_t m_NextBlockId;
    bool m_IncrementalSort = true;

    void SetIncrementalSort(bool val) { m_IncrementalSort = val; }

    VkDeviceSize CalcMaxBlockSize() const;
    // Finds and removes given block from vector.
    void Remove(VmaDeviceMemoryBlock* pBlock);
    // Performs single step in sorting m_Blocks. They may not be fully sorted
    // after this call.
    void IncrementallySortBlocks();
    void SortByFreeSize();

    VkResult AllocatePage(
        VkDeviceSize size,
        VkDeviceSize alignment,
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        VmaAllocation* pAllocation);

    VkResult AllocateFromBlock(
        VmaDeviceMemoryBlock* pBlock,
        VkDeviceSize size,
        VkDeviceSize alignment,
        VmaAllocationCreateFlags allocFlags,
        void* pUserData,
        VmaSuballocationType suballocType,
        uint32_t strategy,
        VmaAllocation* pAllocation);

    VkResult CommitAllocationRequest(
        VmaAllocationRequest& allocRequest,
        VmaDeviceMemoryBlock* pBlock,
        VkDeviceSize alignment,
        VmaAllocationCreateFlags allocFlags,
        void* pUserData,
        VmaSuballocationType suballocType,
        VmaAllocation* pAllocation);

    VkResult CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex);
    bool HasEmptyBlock();
};
#endif // _VMA_BLOCK_VECTOR

#ifndef _VMA_DEFRAGMENTATION_CONTEXT
struct VmaDefragmentationContext_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaDefragmentationContext_T)
public:
    VmaDefragmentationContext_T(
        VmaAllocator hAllocator,
        const VmaDefragmentationInfo& info);
    ~VmaDefragmentationContext_T();

    void GetStats(VmaDefragmentationStats& outStats) { outStats = m_GlobalStats; }

    VkResult DefragmentPassBegin(VmaDefragmentationPassMoveInfo& moveInfo);
    VkResult DefragmentPassEnd(VmaDefragmentationPassMoveInfo& moveInfo);

private:
    // Max number of allocations to ignore due to size constraints before ending single pass
    static const uint8_t MAX_ALLOCS_TO_IGNORE = 16;
    enum class CounterStatus { Pass, Ignore, End };

    struct FragmentedBlock
    {
        uint32_t data;
        VmaDeviceMemoryBlock* block;
    };
    struct StateBalanced
    {
        VkDeviceSize avgFreeSize = 0;
        VkDeviceSize avgAllocSize = UINT64_MAX;
    };
    struct StateExtensive
    {
        enum class Operation : uint8_t
        {
            FindFreeBlockBuffer, FindFreeBlockTexture, FindFreeBlockAll,
            MoveBuffers, MoveTextures, MoveAll,
            Cleanup, Done
        };

        Operation operation = Operation::FindFreeBlockTexture;
        size_t firstFreeBlock = SIZE_MAX;
    };
    struct MoveAllocationData
    {
        VkDeviceSize size;
        VkDeviceSize alignment;
        VmaSuballocationType type;
        VmaAllocationCreateFlags flags;
        VmaDefragmentationMove move = {};
    };

    const VkDeviceSize m_MaxPassBytes;
    const uint32_t m_MaxPassAllocations;

    VmaStlAllocator<VmaDefragmentationMove> m_MoveAllocator;
    VmaVector<VmaDefragmentationMove, VmaStlAllocator<VmaDefragmentationMove>> m_Moves;

    uint8_t m_IgnoredAllocs = 0;
    uint32_t m_Algorithm;
    uint32_t m_BlockVectorCount;
    VmaBlockVector* m_PoolBlockVector;
    VmaBlockVector** m_pBlockVectors;
    size_t m_ImmovableBlockCount = 0;
    VmaDefragmentationStats m_GlobalStats = { 0 };
    VmaDefragmentationStats m_PassStats = { 0 };
    void* m_AlgorithmState = VMA_NULL;

    static MoveAllocationData GetMoveData(VmaAllocHandle handle, VmaBlockMetadata* metadata);
    CounterStatus CheckCounters(VkDeviceSize bytes);
    bool IncrementCounters(VkDeviceSize bytes);
    bool ReallocWithinBlock(VmaBlockVector& vector, VmaDeviceMemoryBlock* block);
    bool AllocInOtherBlock(size_t start, size_t end, MoveAllocationData& data, VmaBlockVector& vector);

    bool ComputeDefragmentation(VmaBlockVector& vector, size_t index);
    bool ComputeDefragmentation_Fast(VmaBlockVector& vector);
    bool ComputeDefragmentation_Balanced(VmaBlockVector& vector, size_t index, bool update);
    bool ComputeDefragmentation_Full(VmaBlockVector& vector);
    bool ComputeDefragmentation_Extensive(VmaBlockVector& vector, size_t index);

    void UpdateVectorStatistics(VmaBlockVector& vector, StateBalanced& state);
    bool MoveDataToFreeBlocks(VmaSuballocationType currentType,
        VmaBlockVector& vector, size_t firstFreeBlock,
        bool& texturePresent, bool& bufferPresent, bool& otherPresent);
};
#endif // _VMA_DEFRAGMENTATION_CONTEXT

#ifndef _VMA_POOL_T
struct VmaPool_T
{
    friend struct VmaPoolListItemTraits;
    VMA_CLASS_NO_COPY_NO_MOVE(VmaPool_T)
public:
    VmaBlockVector m_BlockVector;
    VmaDedicatedAllocationList m_DedicatedAllocations;

    VmaPool_T(
        VmaAllocator hAllocator,
        const VmaPoolCreateInfo& createInfo,
        VkDeviceSize preferredBlockSize);
    ~VmaPool_T();

    uint32_t GetId() const { return m_Id; }
    void SetId(uint32_t id) { VMA_ASSERT(m_Id == 0); m_Id = id; }

    const char* GetName() const { return m_Name; }
    void SetName(const char* pName);

#if VMA_STATS_STRING_ENABLED
    //void PrintDetailedMap(class VmaStringBuilder& sb);
#endif

private:
    uint32_t m_Id;
    char* m_Name;
    VmaPool_T* m_PrevPool = VMA_NULL;
    VmaPool_T* m_NextPool = VMA_NULL;
};

struct VmaPoolListItemTraits
{
    typedef VmaPool_T ItemType;

    static ItemType* GetPrev(const ItemType* item) { return item->m_PrevPool; }
    static ItemType* GetNext(const ItemType* item) { return item->m_NextPool; }
    static ItemType*& AccessPrev(ItemType* item) { return item->m_PrevPool; }
    static ItemType*& AccessNext(ItemType* item) { return item->m_NextPool; }
};
#endif // _VMA_POOL_T

#ifndef _VMA_CURRENT_BUDGET_DATA
struct VmaCurrentBudgetData
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaCurrentBudgetData)
public:

    VMA_ATOMIC_UINT32 m_BlockCount[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT32 m_AllocationCount[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT64 m_BlockBytes[VK_MAX_MEMORY_HEAPS];
    VMA_ATOMIC_UINT64 m_AllocationBytes[VK_MAX_MEMORY_HEAPS];

#if VMA_MEMORY_BUDGET
    VMA_ATOMIC_UINT32 m_OperationsSinceBudgetFetch;
    VMA_RW_MUTEX m_BudgetMutex;
    uint64_t m_VulkanUsage[VK_MAX_MEMORY_HEAPS];
    uint64_t m_VulkanBudget[VK_MAX_MEMORY_HEAPS];
    uint64_t m_BlockBytesAtBudgetFetch[VK_MAX_MEMORY_HEAPS];
#endif // VMA_MEMORY_BUDGET

    VmaCurrentBudgetData();

    void AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
    void RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize);
};

#ifndef _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
VmaCurrentBudgetData::VmaCurrentBudgetData()
{
    for (uint32_t heapIndex = 0; heapIndex < VK_MAX_MEMORY_HEAPS; ++heapIndex)
    {
        m_BlockCount[heapIndex] = 0;
        m_AllocationCount[heapIndex] = 0;
        m_BlockBytes[heapIndex] = 0;
        m_AllocationBytes[heapIndex] = 0;
#if VMA_MEMORY_BUDGET
        m_VulkanUsage[heapIndex] = 0;
        m_VulkanBudget[heapIndex] = 0;
        m_BlockBytesAtBudgetFetch[heapIndex] = 0;
#endif
    }

#if VMA_MEMORY_BUDGET
    m_OperationsSinceBudgetFetch = 0;
#endif
}

void VmaCurrentBudgetData::AddAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    m_AllocationBytes[heapIndex] += allocationSize;
    ++m_AllocationCount[heapIndex];
#if VMA_MEMORY_BUDGET
    ++m_OperationsSinceBudgetFetch;
#endif
}

void VmaCurrentBudgetData::RemoveAllocation(uint32_t heapIndex, VkDeviceSize allocationSize)
{
    VMA_ASSERT(m_AllocationBytes[heapIndex] >= allocationSize);
    m_AllocationBytes[heapIndex] -= allocationSize;
    VMA_ASSERT(m_AllocationCount[heapIndex] > 0);
    --m_AllocationCount[heapIndex];
#if VMA_MEMORY_BUDGET
    ++m_OperationsSinceBudgetFetch;
#endif
}
#endif // _VMA_CURRENT_BUDGET_DATA_FUNCTIONS
#endif // _VMA_CURRENT_BUDGET_DATA

#ifndef _VMA_ALLOCATION_OBJECT_ALLOCATOR
/*
Thread-safe wrapper over VmaPoolAllocator free list, for allocation of VmaAllocation_T objects.
*/
class VmaAllocationObjectAllocator
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocationObjectAllocator)
public:
    VmaAllocationObjectAllocator(const VkAllocationCallbacks* pAllocationCallbacks)
        : m_Allocator(pAllocationCallbacks, 1024) {}

    template<typename... Types> VmaAllocation Allocate(Types&&... args);
    void Free(VmaAllocation hAlloc);

private:
    VMA_MUTEX m_Mutex;
    VmaPoolAllocator<VmaAllocation_T> m_Allocator;
};

template<typename... Types>
VmaAllocation VmaAllocationObjectAllocator::Allocate(Types&&... args)
{
    VmaMutexLock mutexLock(m_Mutex);
    return m_Allocator.Alloc<Types...>(std::forward<Types>(args)...);
}

void VmaAllocationObjectAllocator::Free(VmaAllocation hAlloc)
{
    VmaMutexLock mutexLock(m_Mutex);
    m_Allocator.Free(hAlloc);
}
#endif // _VMA_ALLOCATION_OBJECT_ALLOCATOR

#ifndef _VMA_VIRTUAL_BLOCK_T
struct VmaVirtualBlock_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaVirtualBlock_T)
public:
    const bool m_AllocationCallbacksSpecified;
    const VkAllocationCallbacks m_AllocationCallbacks;

    VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo);
    ~VmaVirtualBlock_T();

    VkResult Init() { return VK_SUCCESS; }
    bool IsEmpty() const { return m_Metadata->IsEmpty(); }
    void Free(VmaVirtualAllocation allocation) { m_Metadata->Free((VmaAllocHandle)allocation); }
    void SetAllocationUserData(VmaVirtualAllocation allocation, void* userData) { m_Metadata->SetAllocationUserData((VmaAllocHandle)allocation, userData); }
    void Clear() { m_Metadata->Clear(); }

    const VkAllocationCallbacks* GetAllocationCallbacks() const;
    void GetAllocationInfo(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo);
    VkResult Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
        VkDeviceSize* outOffset);
    void GetStatistics(VmaStatistics& outStats) const;
    void CalculateDetailedStatistics(VmaDetailedStatistics& outStats) const;
#if VMA_STATS_STRING_ENABLED
    void BuildStatsString(bool detailedMap, VmaStringBuilder& sb) const;
#endif

private:
    VmaBlockMetadata* m_Metadata;
};

#ifndef _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
VmaVirtualBlock_T::VmaVirtualBlock_T(const VmaVirtualBlockCreateInfo& createInfo)
    : m_AllocationCallbacksSpecified(createInfo.pAllocationCallbacks != VMA_NULL),
    m_AllocationCallbacks(createInfo.pAllocationCallbacks != VMA_NULL ? *createInfo.pAllocationCallbacks : VmaEmptyAllocationCallbacks)
{
    const uint32_t algorithm = createInfo.flags & VMA_VIRTUAL_BLOCK_CREATE_ALGORITHM_MASK;
    switch (algorithm)
    {
    case 0:
        m_Metadata = vma_new(GetAllocationCallbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
        break;
    case VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT:
        m_Metadata = vma_new(GetAllocationCallbacks(), VmaBlockMetadata_Linear)(VK_NULL_HANDLE, 1, true);
        break;
    default:
        VMA_ASSERT(0);
        m_Metadata = vma_new(GetAllocationCallbacks(), VmaBlockMetadata_TLSF)(VK_NULL_HANDLE, 1, true);
    }

    m_Metadata->Init(createInfo.size);
}

VmaVirtualBlock_T::~VmaVirtualBlock_T()
{
    // Define macro VMA_DEBUG_LOG_FORMAT to receive the list of the unfreed allocations
    if (!m_Metadata->IsEmpty())
        m_Metadata->DebugLogAllAllocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased virtual allocations.
    VMA_ASSERT(m_Metadata->IsEmpty() && "Some virtual allocations were not freed before destruction of this virtual block!");

    vma_delete(GetAllocationCallbacks(), m_Metadata);
}

const VkAllocationCallbacks* VmaVirtualBlock_T::GetAllocationCallbacks() const
{
    return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : VMA_NULL;
}

void VmaVirtualBlock_T::GetAllocationInfo(VmaVirtualAllocation allocation, VmaVirtualAllocationInfo& outInfo)
{
    m_Metadata->GetAllocationInfo((VmaAllocHandle)allocation, outInfo);
}

VkResult VmaVirtualBlock_T::Allocate(const VmaVirtualAllocationCreateInfo& createInfo, VmaVirtualAllocation& outAllocation,
    VkDeviceSize* outOffset)
{
    VmaAllocationRequest request = {};
    if (m_Metadata->CreateAllocationRequest(
        createInfo.size, // allocSize
        VMA_MAX(createInfo.alignment, (VkDeviceSize)1), // allocAlignment
        (createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0, // upperAddress
        VMA_SUBALLOCATION_TYPE_UNKNOWN, // allocType - unimportant
        createInfo.flags & VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MASK, // strategy
        &request))
    {
        m_Metadata->Alloc(request,
            VMA_SUBALLOCATION_TYPE_UNKNOWN, // type - unimportant
            createInfo.pUserData);
        outAllocation = (VmaVirtualAllocation)request.allocHandle;
        if(outOffset)
            *outOffset = m_Metadata->GetAllocationOffset(request.allocHandle);
        return VK_SUCCESS;
    }
    outAllocation = (VmaVirtualAllocation)VK_NULL_HANDLE;
    if (outOffset)
        *outOffset = UINT64_MAX;
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaVirtualBlock_T::GetStatistics(VmaStatistics& outStats) const
{
    VmaClearStatistics(outStats);
    m_Metadata->AddStatistics(outStats);
}

void VmaVirtualBlock_T::CalculateDetailedStatistics(VmaDetailedStatistics& outStats) const
{
    VmaClearDetailedStatistics(outStats);
    m_Metadata->AddDetailedStatistics(outStats);
}

#if VMA_STATS_STRING_ENABLED
void VmaVirtualBlock_T::BuildStatsString(bool detailedMap, VmaStringBuilder& sb) const
{
    VmaJsonWriter json(GetAllocationCallbacks(), sb);
    json.BeginObject();

    VmaDetailedStatistics stats;
    CalculateDetailedStatistics(stats);

    json.WriteString("Stats");
    VmaPrintDetailedStatistics(json, stats);

    if (detailedMap)
    {
        json.WriteString("Details");
        json.BeginObject();
        m_Metadata->PrintDetailedMap(json);
        json.EndObject();
    }

    json.EndObject();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_VIRTUAL_BLOCK_T_FUNCTIONS
#endif // _VMA_VIRTUAL_BLOCK_T


// Main allocator object.
struct VmaAllocator_T
{
    VMA_CLASS_NO_COPY_NO_MOVE(VmaAllocator_T)
public:
    bool m_UseMutex;
    uint32_t m_VulkanApiVersion;
    bool m_UseKhrDedicatedAllocation; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseKhrBindMemory2; // Can be set only if m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0).
    bool m_UseExtMemoryBudget;
    bool m_UseAmdDeviceCoherentMemory;
    bool m_UseKhrBufferDeviceAddress;
    bool m_UseExtMemoryPriority;
    VkDevice m_hDevice;
    VkInstance m_hInstance;
    bool m_AllocationCallbacksSpecified;
    VkAllocationCallbacks m_AllocationCallbacks;
    VmaDeviceMemoryCallbacks m_DeviceMemoryCallbacks;
    VmaAllocationObjectAllocator m_AllocationObjectAllocator;

    // Each bit (1 << i) is set if HeapSizeLimit is enabled for that heap, so cannot allocate more than the heap size.
    uint32_t m_HeapSizeLimitMask;

    VkPhysicalDeviceProperties m_PhysicalDeviceProperties;
    VkPhysicalDeviceMemoryProperties m_MemProps;

    // Default pools.
    VmaBlockVector* m_pBlockVectors[VK_MAX_MEMORY_TYPES];
    VmaDedicatedAllocationList m_DedicatedAllocations[VK_MAX_MEMORY_TYPES];

    VmaCurrentBudgetData m_Budget;
    VMA_ATOMIC_UINT32 m_DeviceMemoryCount; // Total number of VkDeviceMemory objects.

    VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo);
    VkResult Init(const VmaAllocatorCreateInfo* pCreateInfo);
    ~VmaAllocator_T();

    const VkAllocationCallbacks* GetAllocationCallbacks() const
    {
        return m_AllocationCallbacksSpecified ? &m_AllocationCallbacks : VMA_NULL;
    }
    const VmaVulkanFunctions& GetVulkanFunctions() const
    {
        return m_VulkanFunctions;
    }

    VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }

    VkDeviceSize GetBufferImageGranularity() const
    {
        return VMA_MAX(
            static_cast<VkDeviceSize>(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY),
            m_PhysicalDeviceProperties.limits.bufferImageGranularity);
    }

    uint32_t GetMemoryHeapCount() const { return m_MemProps.memoryHeapCount; }
    uint32_t GetMemoryTypeCount() const { return m_MemProps.memoryTypeCount; }

    uint32_t MemoryTypeIndexToHeapIndex(uint32_t memTypeIndex) const
    {
        VMA_ASSERT(memTypeIndex < m_MemProps.memoryTypeCount);
        return m_MemProps.memoryTypes[memTypeIndex].heapIndex;
    }
    // True when specific memory type is HOST_VISIBLE but not HOST_COHERENT.
    bool IsMemoryTypeNonCoherent(uint32_t memTypeIndex) const
    {
        return (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    }
    // Minimum alignment for all allocations in specific memory type.
    VkDeviceSize GetMemoryTypeMinAlignment(uint32_t memTypeIndex) const
    {
        return IsMemoryTypeNonCoherent(memTypeIndex) ?
            VMA_MAX((VkDeviceSize)VMA_MIN_ALIGNMENT, m_PhysicalDeviceProperties.limits.nonCoherentAtomSize) :
            (VkDeviceSize)VMA_MIN_ALIGNMENT;
    }

    bool IsIntegratedGpu() const
    {
        return m_PhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

    uint32_t GetGlobalMemoryTypeBits() const { return m_GlobalMemoryTypeBits; }

    void GetBufferMemoryRequirements(
        VkBuffer hBuffer,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    void GetImageMemoryRequirements(
        VkImage hImage,
        VkMemoryRequirements& memReq,
        bool& requiresDedicatedAllocation,
        bool& prefersDedicatedAllocation) const;
    VkResult FindMemoryTypeIndex(
        uint32_t memoryTypeBits,
        const VmaAllocationCreateInfo* pAllocationCreateInfo,
        VkFlags bufImgUsage, // VkBufferCreateInfo::usage or VkImageCreateInfo::usage. UINT32_MAX if unknown.
        uint32_t* pMemoryTypeIndex) const;

    // Main allocation function.
    VkResult AllocateMemory(
        const VkMemoryRequirements& vkMemReq,
        bool requiresDedicatedAllocation,
        bool prefersDedicatedAllocation,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VkFlags dedicatedBufferImageUsage, // UINT32_MAX if unknown.
        const VmaAllocationCreateInfo& createInfo,
        VmaSuballocationType suballocType,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Main deallocation function.
    void FreeMemory(
        size_t allocationCount,
        const VmaAllocation* pAllocations);

    void CalculateStatistics(VmaTotalStatistics* pStats);

    void GetHeapBudgets(
        VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount);

#if VMA_STATS_STRING_ENABLED
    void PrintDetailedMap(class VmaJsonWriter& json);
#endif

    void GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo);

    VkResult CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool);
    void DestroyPool(VmaPool pool);
    void GetPoolStatistics(VmaPool pool, VmaStatistics* pPoolStats);
    void CalculatePoolStatistics(VmaPool pool, VmaDetailedStatistics* pPoolStats);

    void SetCurrentFrameIndex(uint32_t frameIndex);
    uint32_t GetCurrentFrameIndex() const { return m_CurrentFrameIndex.load(); }

    VkResult CheckPoolCorruption(VmaPool hPool);
    VkResult CheckCorruption(uint32_t memoryTypeBits);

    // Call to Vulkan function vkAllocateMemory with accompanying bookkeeping.
    VkResult AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory);
    // Call to Vulkan function vkFreeMemory with accompanying bookkeeping.
    void FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory);
    // Call to Vulkan function vkBindBufferMemory or vkBindBufferMemory2KHR.
    VkResult BindVulkanBuffer(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkBuffer buffer,
        const void* pNext);
    // Call to Vulkan function vkBindImageMemory or vkBindImageMemory2KHR.
    VkResult BindVulkanImage(
        VkDeviceMemory memory,
        VkDeviceSize memoryOffset,
        VkImage image,
        const void* pNext);

    VkResult Map(VmaAllocation hAllocation, void** ppData);
    void Unmap(VmaAllocation hAllocation);

    VkResult BindBufferMemory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkBuffer hBuffer,
        const void* pNext);
    VkResult BindImageMemory(
        VmaAllocation hAllocation,
        VkDeviceSize allocationLocalOffset,
        VkImage hImage,
        const void* pNext);

    VkResult FlushOrInvalidateAllocation(
        VmaAllocation hAllocation,
        VkDeviceSize offset, VkDeviceSize size,
        VMA_CACHE_OPERATION op);
    VkResult FlushOrInvalidateAllocations(
        uint32_t allocationCount,
        const VmaAllocation* allocations,
        const VkDeviceSize* offsets, const VkDeviceSize* sizes,
        VMA_CACHE_OPERATION op);

    void FillAllocation(const VmaAllocation hAllocation, uint8_t pattern);

    /*
    Returns bit mask of memory types that can support defragmentation on GPU as
    they support creation of required buffer for copy operations.
    */
    uint32_t GetGpuDefragmentationMemoryTypeBits();

#if VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR GetExternalMemoryHandleTypeFlags(uint32_t memTypeIndex) const
    {
        return m_TypeExternalMemoryHandleTypes[memTypeIndex];
    }
#endif // #if VMA_EXTERNAL_MEMORY

private:
    VkDeviceSize m_PreferredLargeHeapBlockSize;

    VkPhysicalDevice m_PhysicalDevice;
    VMA_ATOMIC_UINT32 m_CurrentFrameIndex;
    VMA_ATOMIC_UINT32 m_GpuDefragmentationMemoryTypeBits; // UINT32_MAX means uninitialized.
#if VMA_EXTERNAL_MEMORY
    VkExternalMemoryHandleTypeFlagsKHR m_TypeExternalMemoryHandleTypes[VK_MAX_MEMORY_TYPES];
#endif // #if VMA_EXTERNAL_MEMORY

    VMA_RW_MUTEX m_PoolsMutex;
    typedef VmaIntrusiveLinkedList<VmaPoolListItemTraits> PoolList;
    // Protected by m_PoolsMutex.
    PoolList m_Pools;
    uint32_t m_NextPoolId;

    VmaVulkanFunctions m_VulkanFunctions;

    // Global bit mask AND-ed with any memoryTypeBits to disallow certain memory types.
    uint32_t m_GlobalMemoryTypeBits;

    void ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_STATIC_VULKAN_FUNCTIONS == 1
    void ImportVulkanFunctions_Static();
#endif

    void ImportVulkanFunctions_Custom(const VmaVulkanFunctions* pVulkanFunctions);

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    void ImportVulkanFunctions_Dynamic();
#endif

    void ValidateVulkanFunctions();

    VkDeviceSize CalcPreferredBlockSize(uint32_t memTypeIndex);

    VkResult AllocateMemoryOfType(
        VmaPool pool,
        VkDeviceSize size,
        VkDeviceSize alignment,
        bool dedicatedPreferred,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VkFlags dedicatedBufferImageUsage,
        const VmaAllocationCreateInfo& createInfo,
        uint32_t memTypeIndex,
        VmaSuballocationType suballocType,
        VmaDedicatedAllocationList& dedicatedAllocations,
        VmaBlockVector& blockVector,
        size_t allocationCount,
        VmaAllocation* pAllocations);

    // Helper function only to be used inside AllocateDedicatedMemory.
    VkResult AllocateDedicatedMemoryPage(
        VmaPool pool,
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        uint32_t memTypeIndex,
        const VkMemoryAllocateInfo& allocInfo,
        bool map,
        bool isUserDataString,
        bool isMappingAllowed,
        void* pUserData,
        VmaAllocation* pAllocation);

    // Allocates and registers new VkDeviceMemory specifically for dedicated allocations.
    VkResult AllocateDedicatedMemory(
        VmaPool pool,
        VkDeviceSize size,
        VmaSuballocationType suballocType,
        VmaDedicatedAllocationList& dedicatedAllocations,
        uint32_t memTypeIndex,
        bool map,
        bool isUserDataString,
        bool isMappingAllowed,
        bool canAliasMemory,
        void* pUserData,
        float priority,
        VkBuffer dedicatedBuffer,
        VkImage dedicatedImage,
        VkFlags dedicatedBufferImageUsage,
        size_t allocationCount,
        VmaAllocation* pAllocations,
        const void* pNextChain = nullptr);

    void FreeDedicatedMemory(const VmaAllocation allocation);

    VkResult CalcMemTypeParams(
        VmaAllocationCreateInfo& outCreateInfo,
        uint32_t memTypeIndex,
        VkDeviceSize size,
        size_t allocationCount);
    VkResult CalcAllocationParams(
        VmaAllocationCreateInfo& outCreateInfo,
        bool dedicatedRequired,
        bool dedicatedPreferred);

    /*
    Calculates and returns bit mask of memory types that can support defragmentation
    on GPU as they support creation of required buffer for copy operations.
    */
    uint32_t CalculateGpuDefragmentationMemoryTypeBits() const;
    uint32_t CalculateGlobalMemoryTypeBits() const;

    bool GetFlushOrInvalidateRange(
        VmaAllocation allocation,
        VkDeviceSize offset, VkDeviceSize size,
        VkMappedMemoryRange& outRange) const;

#if VMA_MEMORY_BUDGET
    void UpdateVulkanBudget();
#endif // #if VMA_MEMORY_BUDGET
};


#ifndef _VMA_MEMORY_FUNCTIONS
static void* VmaMalloc(VmaAllocator hAllocator, size_t size, size_t alignment)
{
    return VmaMalloc(&hAllocator->m_AllocationCallbacks, size, alignment);
}

static void VmaFree(VmaAllocator hAllocator, void* ptr)
{
    VmaFree(&hAllocator->m_AllocationCallbacks, ptr);
}

template<typename T>
static T* VmaAllocate(VmaAllocator hAllocator)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T), VMA_ALIGN_OF(T));
}

template<typename T>
static T* VmaAllocateArray(VmaAllocator hAllocator, size_t count)
{
    return (T*)VmaMalloc(hAllocator, sizeof(T) * count, VMA_ALIGN_OF(T));
}

template<typename T>
static void vma_delete(VmaAllocator hAllocator, T* ptr)
{
    if(ptr != VMA_NULL)
    {
        ptr->~T();
        VmaFree(hAllocator, ptr);
    }
}

template<typename T>
static void vma_delete_array(VmaAllocator hAllocator, T* ptr, size_t count)
{
    if(ptr != VMA_NULL)
    {
        for(size_t i = count; i--; )
            ptr[i].~T();
        VmaFree(hAllocator, ptr);
    }
}
#endif // _VMA_MEMORY_FUNCTIONS

#ifndef _VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS
VmaDeviceMemoryBlock::VmaDeviceMemoryBlock(VmaAllocator hAllocator)
    : m_pMetadata(VMA_NULL),
    m_MemoryTypeIndex(UINT32_MAX),
    m_Id(0),
    m_hMemory(VK_NULL_HANDLE),
    m_MapCount(0),
    m_pMappedData(VMA_NULL) {}

VmaDeviceMemoryBlock::~VmaDeviceMemoryBlock()
{
    VMA_ASSERT(m_MapCount == 0 && "VkDeviceMemory block is being destroyed while it is still mapped.");
    VMA_ASSERT(m_hMemory == VK_NULL_HANDLE);
}

void VmaDeviceMemoryBlock::Init(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t newMemoryTypeIndex,
    VkDeviceMemory newMemory,
    VkDeviceSize newSize,
    uint32_t id,
    uint32_t algorithm,
    VkDeviceSize bufferImageGranularity)
{
    VMA_ASSERT(m_hMemory == VK_NULL_HANDLE);

    m_hParentPool = hParentPool;
    m_MemoryTypeIndex = newMemoryTypeIndex;
    m_Id = id;
    m_hMemory = newMemory;

    switch (algorithm)
    {
    case 0:
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_TLSF)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    case VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT:
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_Linear)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
        break;
    default:
        VMA_ASSERT(0);
        m_pMetadata = vma_new(hAllocator, VmaBlockMetadata_TLSF)(hAllocator->GetAllocationCallbacks(),
            bufferImageGranularity, false); // isVirtual
    }
    m_pMetadata->Init(newSize);
}

void VmaDeviceMemoryBlock::Destroy(VmaAllocator allocator)
{
    // Define macro VMA_DEBUG_LOG_FORMAT to receive the list of the unfreed allocations
    if (!m_pMetadata->IsEmpty())
        m_pMetadata->DebugLogAllAllocations();
    // This is the most important assert in the entire library.
    // Hitting it means you have some memory leak - unreleased VmaAllocation objects.
    VMA_ASSERT(m_pMetadata->IsEmpty() && "Some allocations were not freed before destruction of this memory block!");

    VMA_ASSERT(m_hMemory != VK_NULL_HANDLE);
    allocator->FreeVulkanMemory(m_MemoryTypeIndex, m_pMetadata->GetSize(), m_hMemory);
    m_hMemory = VK_NULL_HANDLE;

    vma_delete(allocator, m_pMetadata);
    m_pMetadata = VMA_NULL;
}

void VmaDeviceMemoryBlock::PostAlloc(VmaAllocator hAllocator)
{
    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    m_MappingHysteresis.PostAlloc();
}

void VmaDeviceMemoryBlock::PostFree(VmaAllocator hAllocator)
{
    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    if(m_MappingHysteresis.PostFree())
    {
        VMA_ASSERT(m_MappingHysteresis.GetExtraMapping() == 0);
        if (m_MapCount == 0)
        {
            m_pMappedData = VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(hAllocator->m_hDevice, m_hMemory);
        }
    }
}

bool VmaDeviceMemoryBlock::Validate() const
{
    VMA_VALIDATE((m_hMemory != VK_NULL_HANDLE) &&
        (m_pMetadata->GetSize() != 0));

    return m_pMetadata->Validate();
}

VkResult VmaDeviceMemoryBlock::CheckCorruption(VmaAllocator hAllocator)
{
    void* pData = nullptr;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    res = m_pMetadata->CheckCorruption(pData);

    Unmap(hAllocator, 1);

    return res;
}

VkResult VmaDeviceMemoryBlock::Map(VmaAllocator hAllocator, uint32_t count, void** ppData)
{
    if (count == 0)
    {
        return VK_SUCCESS;
    }

    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    const uint32_t oldTotalMapCount = m_MapCount + m_MappingHysteresis.GetExtraMapping();
    m_MappingHysteresis.PostMap();
    if (oldTotalMapCount != 0)
    {
        m_MapCount += count;
        VMA_ASSERT(m_pMappedData != VMA_NULL);
        if (ppData != VMA_NULL)
        {
            *ppData = m_pMappedData;
        }
        return VK_SUCCESS;
    }
    else
    {
        VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
            hAllocator->m_hDevice,
            m_hMemory,
            0, // offset
            VK_WHOLE_SIZE,
            0, // flags
            &m_pMappedData);
        if (result == VK_SUCCESS)
        {
            if (ppData != VMA_NULL)
            {
                *ppData = m_pMappedData;
            }
            m_MapCount = count;
        }
        return result;
    }
}

void VmaDeviceMemoryBlock::Unmap(VmaAllocator hAllocator, uint32_t count)
{
    if (count == 0)
    {
        return;
    }

    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    if (m_MapCount >= count)
    {
        m_MapCount -= count;
        const uint32_t totalMapCount = m_MapCount + m_MappingHysteresis.GetExtraMapping();
        if (totalMapCount == 0)
        {
            m_pMappedData = VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(hAllocator->m_hDevice, m_hMemory);
        }
        m_MappingHysteresis.PostUnmap();
    }
    else
    {
        VMA_ASSERT(0 && "VkDeviceMemory block is being unmapped while it was not previously mapped.");
    }
}

VkResult VmaDeviceMemoryBlock::WriteMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);

    void* pData;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    VmaWriteMagicValue(pData, allocOffset + allocSize);

    Unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::ValidateMagicValueAfterAllocation(VmaAllocator hAllocator, VkDeviceSize allocOffset, VkDeviceSize allocSize)
{
    VMA_ASSERT(VMA_DEBUG_MARGIN > 0 && VMA_DEBUG_MARGIN % 4 == 0 && VMA_DEBUG_DETECT_CORRUPTION);

    void* pData;
    VkResult res = Map(hAllocator, 1, &pData);
    if (res != VK_SUCCESS)
    {
        return res;
    }

    if (!VmaValidateMagicValue(pData, allocOffset + allocSize))
    {
        VMA_ASSERT(0 && "MEMORY CORRUPTION DETECTED AFTER FREED ALLOCATION!");
    }

    Unmap(hAllocator, 1);
    return VK_SUCCESS;
}

VkResult VmaDeviceMemoryBlock::BindBufferMemory(
    const VmaAllocator hAllocator,
    const VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanBuffer(m_hMemory, memoryOffset, hBuffer, pNext);
}

VkResult VmaDeviceMemoryBlock::BindImageMemory(
    const VmaAllocator hAllocator,
    const VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VMA_ASSERT(hAllocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_BLOCK &&
        hAllocation->GetBlock() == this);
    VMA_ASSERT(allocationLocalOffset < hAllocation->GetSize() &&
        "Invalid allocationLocalOffset. Did you forget that this offset is relative to the beginning of the allocation, not the whole memory block?");
    const VkDeviceSize memoryOffset = hAllocation->GetOffset() + allocationLocalOffset;
    // This lock is important so that we don't call vkBind... and/or vkMap... simultaneously on the same VkDeviceMemory from multiple threads.
    VmaMutexLock lock(m_MapAndBindMutex, hAllocator->m_UseMutex);
    return hAllocator->BindVulkanImage(m_hMemory, memoryOffset, hImage, pNext);
}
#endif // _VMA_DEVICE_MEMORY_BLOCK_FUNCTIONS

#ifndef _VMA_ALLOCATION_T_FUNCTIONS
VmaAllocation_T::VmaAllocation_T(bool mappingAllowed)
    : m_Alignment{ 1 },
    m_Size{ 0 },
    m_pUserData{ VMA_NULL },
    m_pName{ VMA_NULL },
    m_MemoryTypeIndex{ 0 },
    m_Type{ (uint8_t)ALLOCATION_TYPE_NONE },
    m_SuballocationType{ (uint8_t)VMA_SUBALLOCATION_TYPE_UNKNOWN },
    m_MapCount{ 0 },
    m_Flags{ 0 }
{
    if(mappingAllowed)
        m_Flags |= (uint8_t)FLAG_MAPPING_ALLOWED;

#if VMA_STATS_STRING_ENABLED
    m_BufferImageUsage = 0;
#endif
}

VmaAllocation_T::~VmaAllocation_T()
{
    VMA_ASSERT(m_MapCount == 0 && "Allocation was not unmapped before destruction.");

    // Check if owned string was freed.
    VMA_ASSERT(m_pName == VMA_NULL);
}

void VmaAllocation_T::InitBlockAllocation(
    VmaDeviceMemoryBlock* block,
    VmaAllocHandle allocHandle,
    VkDeviceSize alignment,
    VkDeviceSize size,
    uint32_t memoryTypeIndex,
    VmaSuballocationType suballocationType,
    bool mapped)
{
    VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
    VMA_ASSERT(block != VMA_NULL);
    m_Type = (uint8_t)ALLOCATION_TYPE_BLOCK;
    m_Alignment = alignment;
    m_Size = size;
    m_MemoryTypeIndex = memoryTypeIndex;
    if(mapped)
    {
        VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        m_Flags |= (uint8_t)FLAG_PERSISTENT_MAP;
    }
    m_SuballocationType = (uint8_t)suballocationType;
    m_BlockAllocation.m_Block = block;
    m_BlockAllocation.m_AllocHandle = allocHandle;
}

void VmaAllocation_T::InitDedicatedAllocation(
    VmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceMemory hMemory,
    VmaSuballocationType suballocationType,
    void* pMappedData,
    VkDeviceSize size)
{
    VMA_ASSERT(m_Type == ALLOCATION_TYPE_NONE);
    VMA_ASSERT(hMemory != VK_NULL_HANDLE);
    m_Type = (uint8_t)ALLOCATION_TYPE_DEDICATED;
    m_Alignment = 0;
    m_Size = size;
    m_MemoryTypeIndex = memoryTypeIndex;
    m_SuballocationType = (uint8_t)suballocationType;
    if(pMappedData != VMA_NULL)
    {
        VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");
        m_Flags |= (uint8_t)FLAG_PERSISTENT_MAP;
    }
    m_DedicatedAllocation.m_hParentPool = hParentPool;
    m_DedicatedAllocation.m_hMemory = hMemory;
    m_DedicatedAllocation.m_pMappedData = pMappedData;
    m_DedicatedAllocation.m_Prev = VMA_NULL;
    m_DedicatedAllocation.m_Next = VMA_NULL;
}

void VmaAllocation_T::SetName(VmaAllocator hAllocator, const char* pName)
{
    VMA_ASSERT(pName == VMA_NULL || pName != m_pName);

    FreeName(hAllocator);

    if (pName != VMA_NULL)
        m_pName = VmaCreateStringCopy(hAllocator->GetAllocationCallbacks(), pName);
}

uint8_t VmaAllocation_T::SwapBlockAllocation(VmaAllocator hAllocator, VmaAllocation allocation)
{
    VMA_ASSERT(allocation != VMA_NULL);
    VMA_ASSERT(m_Type == ALLOCATION_TYPE_BLOCK);
    VMA_ASSERT(allocation->m_Type == ALLOCATION_TYPE_BLOCK);

    if (m_MapCount != 0)
        m_BlockAllocation.m_Block->Unmap(hAllocator, m_MapCount);

    m_BlockAllocation.m_Block->m_pMetadata->SetAllocationUserData(m_BlockAllocation.m_AllocHandle, allocation);
    VMA_SWAP(m_BlockAllocation, allocation->m_BlockAllocation);
    m_BlockAllocation.m_Block->m_pMetadata->SetAllocationUserData(m_BlockAllocation.m_AllocHandle, this);

#if VMA_STATS_STRING_ENABLED
    VMA_SWAP(m_BufferImageUsage, allocation->m_BufferImageUsage);
#endif
    return m_MapCount;
}

VmaAllocHandle VmaAllocation_T::GetAllocHandle() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_AllocHandle;
    case ALLOCATION_TYPE_DEDICATED:
        return VK_NULL_HANDLE;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceSize VmaAllocation_T::GetOffset() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->m_pMetadata->GetAllocationOffset(m_BlockAllocation.m_AllocHandle);
    case ALLOCATION_TYPE_DEDICATED:
        return 0;
    default:
        VMA_ASSERT(0);
        return 0;
    }
}

VmaPool VmaAllocation_T::GetParentPool() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->GetParentPool();
    case ALLOCATION_TYPE_DEDICATED:
        return m_DedicatedAllocation.m_hParentPool;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

VkDeviceMemory VmaAllocation_T::GetMemory() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        return m_BlockAllocation.m_Block->GetDeviceMemory();
    case ALLOCATION_TYPE_DEDICATED:
        return m_DedicatedAllocation.m_hMemory;
    default:
        VMA_ASSERT(0);
        return VK_NULL_HANDLE;
    }
}

void* VmaAllocation_T::GetMappedData() const
{
    switch (m_Type)
    {
    case ALLOCATION_TYPE_BLOCK:
        if (m_MapCount != 0 || IsPersistentMap())
        {
            void* pBlockData = m_BlockAllocation.m_Block->GetMappedData();
            VMA_ASSERT(pBlockData != VMA_NULL);
            return (char*)pBlockData + GetOffset();
        }
        else
        {
            return VMA_NULL;
        }
        break;
    case ALLOCATION_TYPE_DEDICATED:
        VMA_ASSERT((m_DedicatedAllocation.m_pMappedData != VMA_NULL) == (m_MapCount != 0 || IsPersistentMap()));
        return m_DedicatedAllocation.m_pMappedData;
    default:
        VMA_ASSERT(0);
        return VMA_NULL;
    }
}

void VmaAllocation_T::BlockAllocMap()
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);
    VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    if (m_MapCount < 0xFF)
    {
        ++m_MapCount;
    }
    else
    {
        VMA_ASSERT(0 && "Allocation mapped too many times simultaneously.");
    }
}

void VmaAllocation_T::BlockAllocUnmap()
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_BLOCK);

    if (m_MapCount > 0)
    {
        --m_MapCount;
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping allocation not previously mapped.");
    }
}

VkResult VmaAllocation_T::DedicatedAllocMap(VmaAllocator hAllocator, void** ppData)
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);
    VMA_ASSERT(IsMappingAllowed() && "Mapping is not allowed on this allocation! Please use one of the new VMA_ALLOCATION_CREATE_HOST_ACCESS_* flags when creating it.");

    if (m_MapCount != 0 || IsPersistentMap())
    {
        if (m_MapCount < 0xFF)
        {
            VMA_ASSERT(m_DedicatedAllocation.m_pMappedData != VMA_NULL);
            *ppData = m_DedicatedAllocation.m_pMappedData;
            ++m_MapCount;
            return VK_SUCCESS;
        }
        else
        {
            VMA_ASSERT(0 && "Dedicated allocation mapped too many times simultaneously.");
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
    }
    else
    {
        VkResult result = (*hAllocator->GetVulkanFunctions().vkMapMemory)(
            hAllocator->m_hDevice,
            m_DedicatedAllocation.m_hMemory,
            0, // offset
            VK_WHOLE_SIZE,
            0, // flags
            ppData);
        if (result == VK_SUCCESS)
        {
            m_DedicatedAllocation.m_pMappedData = *ppData;
            m_MapCount = 1;
        }
        return result;
    }
}

void VmaAllocation_T::DedicatedAllocUnmap(VmaAllocator hAllocator)
{
    VMA_ASSERT(GetType() == ALLOCATION_TYPE_DEDICATED);

    if (m_MapCount > 0)
    {
        --m_MapCount;
        if (m_MapCount == 0 && !IsPersistentMap())
        {
            m_DedicatedAllocation.m_pMappedData = VMA_NULL;
            (*hAllocator->GetVulkanFunctions().vkUnmapMemory)(
                hAllocator->m_hDevice,
                m_DedicatedAllocation.m_hMemory);
        }
    }
    else
    {
        VMA_ASSERT(0 && "Unmapping dedicated allocation not previously mapped.");
    }
}

#if VMA_STATS_STRING_ENABLED
void VmaAllocation_T::InitBufferImageUsage(uint32_t bufferImageUsage)
{
    VMA_ASSERT(m_BufferImageUsage == 0);
    m_BufferImageUsage = bufferImageUsage;
}

void VmaAllocation_T::PrintParameters(class VmaJsonWriter& json) const
{
    json.WriteString("Type");
    json.WriteString(VMA_SUBALLOCATION_TYPE_NAMES[m_SuballocationType]);

    json.WriteString("Size");
    json.WriteNumber(m_Size);
    json.WriteString("Usage");
    json.WriteNumber(m_BufferImageUsage);

    if (m_pUserData != VMA_NULL)
    {
        json.WriteString("CustomData");
        json.BeginString();
        json.ContinueString_Pointer(m_pUserData);
        json.EndString();
    }
    if (m_pName != VMA_NULL)
    {
        json.WriteString("Name");
        json.WriteString(m_pName);
    }
}
#endif // VMA_STATS_STRING_ENABLED

void VmaAllocation_T::FreeName(VmaAllocator hAllocator)
{
    if(m_pName)
    {
        VmaFreeString(hAllocator->GetAllocationCallbacks(), m_pName);
        m_pName = VMA_NULL;
    }
}
#endif // _VMA_ALLOCATION_T_FUNCTIONS

#ifndef _VMA_BLOCK_VECTOR_FUNCTIONS
VmaBlockVector::VmaBlockVector(
    VmaAllocator hAllocator,
    VmaPool hParentPool,
    uint32_t memoryTypeIndex,
    VkDeviceSize preferredBlockSize,
    size_t minBlockCount,
    size_t maxBlockCount,
    VkDeviceSize bufferImageGranularity,
    bool explicitBlockSize,
    uint32_t algorithm,
    float priority,
    VkDeviceSize minAllocationAlignment,
    void* pMemoryAllocateNext)
    : m_hAllocator(hAllocator),
    m_hParentPool(hParentPool),
    m_MemoryTypeIndex(memoryTypeIndex),
    m_PreferredBlockSize(preferredBlockSize),
    m_MinBlockCount(minBlockCount),
    m_MaxBlockCount(maxBlockCount),
    m_BufferImageGranularity(bufferImageGranularity),
    m_ExplicitBlockSize(explicitBlockSize),
    m_Algorithm(algorithm),
    m_Priority(priority),
    m_MinAllocationAlignment(minAllocationAlignment),
    m_pMemoryAllocateNext(pMemoryAllocateNext),
    m_Blocks(VmaStlAllocator<VmaDeviceMemoryBlock*>(hAllocator->GetAllocationCallbacks())),
    m_NextBlockId(0) {}

VmaBlockVector::~VmaBlockVector()
{
    for (size_t i = m_Blocks.size(); i--; )
    {
        m_Blocks[i]->Destroy(m_hAllocator);
        vma_delete(m_hAllocator, m_Blocks[i]);
    }
}

VkResult VmaBlockVector::CreateMinBlocks()
{
    for (size_t i = 0; i < m_MinBlockCount; ++i)
    {
        VkResult res = CreateBlock(m_PreferredBlockSize, VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

void VmaBlockVector::AddStatistics(VmaStatistics& inoutStats)
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    const size_t blockCount = m_Blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->Validate());
        pBlock->m_pMetadata->AddStatistics(inoutStats);
    }
}

void VmaBlockVector::AddDetailedStatistics(VmaDetailedStatistics& inoutStats)
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);

    const size_t blockCount = m_Blocks.size();
    for (uint32_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
    {
        const VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VMA_HEAVY_ASSERT(pBlock->Validate());
        pBlock->m_pMetadata->AddDetailedStatistics(inoutStats);
    }
}

bool VmaBlockVector::IsEmpty()
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    return m_Blocks.empty();
}

bool VmaBlockVector::IsCorruptionDetectionEnabled() const
{
    const uint32_t requiredMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return (VMA_DEBUG_DETECT_CORRUPTION != 0) &&
        (VMA_DEBUG_MARGIN > 0) &&
        (m_Algorithm == 0 || m_Algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT) &&
        (m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags & requiredMemFlags) == requiredMemFlags;
}

VkResult VmaBlockVector::Allocate(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    size_t allocIndex;
    VkResult res = VK_SUCCESS;

    alignment = VMA_MAX(alignment, m_MinAllocationAlignment);

    if (IsCorruptionDetectionEnabled())
    {
        size = VmaAlignUp<VkDeviceSize>(size, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
        alignment = VmaAlignUp<VkDeviceSize>(alignment, sizeof(VMA_CORRUPTION_DETECTION_MAGIC_VALUE));
    }

    {
        VmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);
        for (allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            res = AllocatePage(
                size,
                alignment,
                createInfo,
                suballocType,
                pAllocations + allocIndex);
            if (res != VK_SUCCESS)
            {
                break;
            }
        }
    }

    if (res != VK_SUCCESS)
    {
        // Free all already created allocations.
        while (allocIndex--)
            Free(pAllocations[allocIndex]);
        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaBlockVector::AllocatePage(
    VkDeviceSize size,
    VkDeviceSize alignment,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    VmaAllocation* pAllocation)
{
    const bool isUpperAddress = (createInfo.flags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    VkDeviceSize freeMemory;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        VmaBudget heapBudget = {};
        m_hAllocator->GetHeapBudgets(&heapBudget, heapIndex, 1);
        freeMemory = (heapBudget.usage < heapBudget.budget) ? (heapBudget.budget - heapBudget.usage) : 0;
    }

    const bool canFallbackToDedicated = !HasExplicitBlockSize() &&
        (createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0;
    const bool canCreateNewBlock =
        ((createInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0) &&
        (m_Blocks.size() < m_MaxBlockCount) &&
        (freeMemory >= size || !canFallbackToDedicated);
    uint32_t strategy = createInfo.flags & VMA_ALLOCATION_CREATE_STRATEGY_MASK;

    // Upper address can only be used with linear allocator and within single memory block.
    if (isUpperAddress &&
        (m_Algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT || m_MaxBlockCount > 1))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    // Early reject: requested allocation size is larger that maximum block size for this block vector.
    if (size + VMA_DEBUG_MARGIN > m_PreferredBlockSize)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    // 1. Search existing allocations. Try to allocate.
    if (m_Algorithm == VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Use only last block.
        if (!m_Blocks.empty())
        {
            VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks.back();
            VMA_ASSERT(pCurrBlock);
            VkResult res = AllocateFromBlock(
                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                VMA_DEBUG_LOG_FORMAT("    Returned from last block #%u", pCurrBlock->GetId());
                IncrementallySortBlocks();
                return VK_SUCCESS;
            }
        }
    }
    else
    {
        if (strategy != VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT) // MIN_MEMORY or default
        {
            const bool isHostVisible =
                (m_hAllocator->m_MemProps.memoryTypes[m_MemoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            if(isHostVisible)
            {
                const bool isMappingAllowed = (createInfo.flags &
                    (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;
                /*
                For non-mappable allocations, check blocks that are not mapped first.
                For mappable allocations, check blocks that are already mapped first.
                This way, having many blocks, we will separate mappable and non-mappable allocations,
                hopefully limiting the number of blocks that are mapped, which will help tools like RenderDoc.
                */
                for(size_t mappingI = 0; mappingI < 2; ++mappingI)
                {
                    // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                    for (size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
                    {
                        VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                        VMA_ASSERT(pCurrBlock);
                        const bool isBlockMapped = pCurrBlock->GetMappedData() != VMA_NULL;
                        if((mappingI == 0) == (isMappingAllowed == isBlockMapped))
                        {
                            VkResult res = AllocateFromBlock(
                                pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                            if (res == VK_SUCCESS)
                            {
                                VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%u", pCurrBlock->GetId());
                                IncrementallySortBlocks();
                                return VK_SUCCESS;
                            }
                        }
                    }
                }
            }
            else
            {
                // Forward order in m_Blocks - prefer blocks with smallest amount of free space.
                for (size_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
                {
                    VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                    VMA_ASSERT(pCurrBlock);
                    VkResult res = AllocateFromBlock(
                        pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                    if (res == VK_SUCCESS)
                    {
                        VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%u", pCurrBlock->GetId());
                        IncrementallySortBlocks();
                        return VK_SUCCESS;
                    }
                }
            }
        }
        else // VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT
        {
            // Backward order in m_Blocks - prefer blocks with largest amount of free space.
            for (size_t blockIndex = m_Blocks.size(); blockIndex--; )
            {
                VmaDeviceMemoryBlock* const pCurrBlock = m_Blocks[blockIndex];
                VMA_ASSERT(pCurrBlock);
                VkResult res = AllocateFromBlock(pCurrBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
                if (res == VK_SUCCESS)
                {
                    VMA_DEBUG_LOG_FORMAT("    Returned from existing block #%u", pCurrBlock->GetId());
                    IncrementallySortBlocks();
                    return VK_SUCCESS;
                }
            }
        }
    }

    // 2. Try to create new block.
    if (canCreateNewBlock)
    {
        // Calculate optimal size for new block.
        VkDeviceSize newBlockSize = m_PreferredBlockSize;
        uint32_t newBlockSizeShift = 0;
        const uint32_t NEW_BLOCK_SIZE_SHIFT_MAX = 3;

        if (!m_ExplicitBlockSize)
        {
            // Allocate 1/8, 1/4, 1/2 as first blocks.
            const VkDeviceSize maxExistingBlockSize = CalcMaxBlockSize();
            for (uint32_t i = 0; i < NEW_BLOCK_SIZE_SHIFT_MAX; ++i)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize > maxExistingBlockSize && smallerNewBlockSize >= size * 2)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                }
                else
                {
                    break;
                }
            }
        }

        size_t newBlockIndex = 0;
        VkResult res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
            CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
        // Allocation of this size failed? Try 1/2, 1/4, 1/8 of m_PreferredBlockSize.
        if (!m_ExplicitBlockSize)
        {
            while (res < 0 && newBlockSizeShift < NEW_BLOCK_SIZE_SHIFT_MAX)
            {
                const VkDeviceSize smallerNewBlockSize = newBlockSize / 2;
                if (smallerNewBlockSize >= size)
                {
                    newBlockSize = smallerNewBlockSize;
                    ++newBlockSizeShift;
                    res = (newBlockSize <= freeMemory || !canFallbackToDedicated) ?
                        CreateBlock(newBlockSize, &newBlockIndex) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
                }
                else
                {
                    break;
                }
            }
        }

        if (res == VK_SUCCESS)
        {
            VmaDeviceMemoryBlock* const pBlock = m_Blocks[newBlockIndex];
            VMA_ASSERT(pBlock->m_pMetadata->GetSize() >= size);

            res = AllocateFromBlock(
                pBlock, size, alignment, createInfo.flags, createInfo.pUserData, suballocType, strategy, pAllocation);
            if (res == VK_SUCCESS)
            {
                VMA_DEBUG_LOG_FORMAT("    Created new block #%u Size=%llu", pBlock->GetId(), newBlockSize);
                IncrementallySortBlocks();
                return VK_SUCCESS;
            }
            else
            {
                // Allocation from new block failed, possibly due to VMA_DEBUG_MARGIN or alignment.
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
        }
    }

    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

void VmaBlockVector::Free(const VmaAllocation hAllocation)
{
    VmaDeviceMemoryBlock* pBlockToDelete = VMA_NULL;

    bool budgetExceeded = false;
    {
        const uint32_t heapIndex = m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex);
        VmaBudget heapBudget = {};
        m_hAllocator->GetHeapBudgets(&heapBudget, heapIndex, 1);
        budgetExceeded = heapBudget.usage >= heapBudget.budget;
    }

    // Scope for lock.
    {
        VmaMutexLockWrite lock(m_Mutex, m_hAllocator->m_UseMutex);

        VmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();

        if (IsCorruptionDetectionEnabled())
        {
            VkResult res = pBlock->ValidateMagicValueAfterAllocation(m_hAllocator, hAllocation->GetOffset(), hAllocation->GetSize());
            VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to validate magic value.");
        }

        if (hAllocation->IsPersistentMap())
        {
            pBlock->Unmap(m_hAllocator, 1);
        }

        const bool hadEmptyBlockBeforeFree = HasEmptyBlock();
        pBlock->m_pMetadata->Free(hAllocation->GetAllocHandle());
        pBlock->PostFree(m_hAllocator);
        VMA_HEAVY_ASSERT(pBlock->Validate());

        VMA_DEBUG_LOG_FORMAT("  Freed from MemoryTypeIndex=%u", m_MemoryTypeIndex);

        const bool canDeleteBlock = m_Blocks.size() > m_MinBlockCount;
        // pBlock became empty after this deallocation.
        if (pBlock->m_pMetadata->IsEmpty())
        {
            // Already had empty block. We don't want to have two, so delete this one.
            if ((hadEmptyBlockBeforeFree || budgetExceeded) && canDeleteBlock)
            {
                pBlockToDelete = pBlock;
                Remove(pBlock);
            }
            // else: We now have one empty block - leave it. A hysteresis to avoid allocating whole block back and forth.
        }
        // pBlock didn't become empty, but we have another empty block - find and free that one.
        // (This is optional, heuristics.)
        else if (hadEmptyBlockBeforeFree && canDeleteBlock)
        {
            VmaDeviceMemoryBlock* pLastBlock = m_Blocks.back();
            if (pLastBlock->m_pMetadata->IsEmpty())
            {
                pBlockToDelete = pLastBlock;
                m_Blocks.pop_back();
            }
        }

        IncrementallySortBlocks();
    }

    // Destruction of a free block. Deferred until this point, outside of mutex
    // lock, for performance reason.
    if (pBlockToDelete != VMA_NULL)
    {
        VMA_DEBUG_LOG_FORMAT("    Deleted empty block #%u", pBlockToDelete->GetId());
        pBlockToDelete->Destroy(m_hAllocator);
        vma_delete(m_hAllocator, pBlockToDelete);
    }

    m_hAllocator->m_Budget.RemoveAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), hAllocation->GetSize());
    m_hAllocator->m_AllocationObjectAllocator.Free(hAllocation);
}

VkDeviceSize VmaBlockVector::CalcMaxBlockSize() const
{
    VkDeviceSize result = 0;
    for (size_t i = m_Blocks.size(); i--; )
    {
        result = VMA_MAX(result, m_Blocks[i]->m_pMetadata->GetSize());
        if (result >= m_PreferredBlockSize)
        {
            break;
        }
    }
    return result;
}

void VmaBlockVector::Remove(VmaDeviceMemoryBlock* pBlock)
{
    for (uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        if (m_Blocks[blockIndex] == pBlock)
        {
            VmaVectorRemove(m_Blocks, blockIndex);
            return;
        }
    }
    VMA_ASSERT(0);
}

void VmaBlockVector::IncrementallySortBlocks()
{
    if (!m_IncrementalSort)
        return;
    if (m_Algorithm != VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
    {
        // Bubble sort only until first swap.
        for (size_t i = 1; i < m_Blocks.size(); ++i)
        {
            if (m_Blocks[i - 1]->m_pMetadata->GetSumFreeSize() > m_Blocks[i]->m_pMetadata->GetSumFreeSize())
            {
                VMA_SWAP(m_Blocks[i - 1], m_Blocks[i]);
                return;
            }
        }
    }
}

void VmaBlockVector::SortByFreeSize()
{
    VMA_SORT(m_Blocks.begin(), m_Blocks.end(),
        [](VmaDeviceMemoryBlock* b1, VmaDeviceMemoryBlock* b2) -> bool
        {
            return b1->m_pMetadata->GetSumFreeSize() < b2->m_pMetadata->GetSumFreeSize();
        });
}

VkResult VmaBlockVector::AllocateFromBlock(
    VmaDeviceMemoryBlock* pBlock,
    VkDeviceSize size,
    VkDeviceSize alignment,
    VmaAllocationCreateFlags allocFlags,
    void* pUserData,
    VmaSuballocationType suballocType,
    uint32_t strategy,
    VmaAllocation* pAllocation)
{
    const bool isUpperAddress = (allocFlags & VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT) != 0;

    VmaAllocationRequest currRequest = {};
    if (pBlock->m_pMetadata->CreateAllocationRequest(
        size,
        alignment,
        isUpperAddress,
        suballocType,
        strategy,
        &currRequest))
    {
        return CommitAllocationRequest(currRequest, pBlock, alignment, allocFlags, pUserData, suballocType, pAllocation);
    }
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VkResult VmaBlockVector::CommitAllocationRequest(
    VmaAllocationRequest& allocRequest,
    VmaDeviceMemoryBlock* pBlock,
    VkDeviceSize alignment,
    VmaAllocationCreateFlags allocFlags,
    void* pUserData,
    VmaSuballocationType suballocType,
    VmaAllocation* pAllocation)
{
    const bool mapped = (allocFlags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0;
    const bool isUserDataString = (allocFlags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0;
    const bool isMappingAllowed = (allocFlags &
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0;

    pBlock->PostAlloc(m_hAllocator);
    // Allocate from pCurrBlock.
    if (mapped)
    {
        VkResult res = pBlock->Map(m_hAllocator, 1, VMA_NULL);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }

    *pAllocation = m_hAllocator->m_AllocationObjectAllocator.Allocate(isMappingAllowed);
    pBlock->m_pMetadata->Alloc(allocRequest, suballocType, *pAllocation);
    (*pAllocation)->InitBlockAllocation(
        pBlock,
        allocRequest.allocHandle,
        alignment,
        allocRequest.size, // Not size, as actual allocation size may be larger than requested!
        m_MemoryTypeIndex,
        suballocType,
        mapped);
    VMA_HEAVY_ASSERT(pBlock->Validate());
    if (isUserDataString)
        (*pAllocation)->SetName(m_hAllocator, (const char*)pUserData);
    else
        (*pAllocation)->SetUserData(m_hAllocator, pUserData);
    m_hAllocator->m_Budget.AddAllocation(m_hAllocator->MemoryTypeIndexToHeapIndex(m_MemoryTypeIndex), allocRequest.size);
    if (VMA_DEBUG_INITIALIZE_ALLOCATIONS)
    {
        m_hAllocator->FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
    }
    if (IsCorruptionDetectionEnabled())
    {
        VkResult res = pBlock->WriteMagicValueAfterAllocation(m_hAllocator, (*pAllocation)->GetOffset(), allocRequest.size);
        VMA_ASSERT(res == VK_SUCCESS && "Couldn't map block memory to write magic value.");
    }
    return VK_SUCCESS;
}

VkResult VmaBlockVector::CreateBlock(VkDeviceSize blockSize, size_t* pNewBlockIndex)
{
    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.pNext = m_pMemoryAllocateNext;
    allocInfo.memoryTypeIndex = m_MemoryTypeIndex;
    allocInfo.allocationSize = blockSize;

#if VMA_BUFFER_DEVICE_ADDRESS
    // Every standalone block can potentially contain a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT - always enable the feature.
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if (m_hAllocator->m_UseKhrBufferDeviceAddress)
    {
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        VmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
    }
#endif // VMA_BUFFER_DEVICE_ADDRESS

#if VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if (m_hAllocator->m_UseExtMemoryPriority)
    {
        VMA_ASSERT(m_Priority >= 0.f && m_Priority <= 1.f);
        priorityInfo.priority = m_Priority;
        VmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // VMA_MEMORY_PRIORITY

#if VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = m_hAllocator->GetExternalMemoryHandleTypeFlags(m_MemoryTypeIndex);
    if (exportMemoryAllocInfo.handleTypes != 0)
    {
        VmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // VMA_EXTERNAL_MEMORY

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult res = m_hAllocator->AllocateVulkanMemory(&allocInfo, &mem);
    if (res < 0)
    {
        return res;
    }

    // New VkDeviceMemory successfully created.

    // Create new Allocation for it.
    VmaDeviceMemoryBlock* const pBlock = vma_new(m_hAllocator, VmaDeviceMemoryBlock)(m_hAllocator);
    pBlock->Init(
        m_hAllocator,
        m_hParentPool,
        m_MemoryTypeIndex,
        mem,
        allocInfo.allocationSize,
        m_NextBlockId++,
        m_Algorithm,
        m_BufferImageGranularity);

    m_Blocks.push_back(pBlock);
    if (pNewBlockIndex != VMA_NULL)
    {
        *pNewBlockIndex = m_Blocks.size() - 1;
    }

    return VK_SUCCESS;
}

bool VmaBlockVector::HasEmptyBlock()
{
    for (size_t index = 0, count = m_Blocks.size(); index < count; ++index)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[index];
        if (pBlock->m_pMetadata->IsEmpty())
        {
            return true;
        }
    }
    return false;
}

#if VMA_STATS_STRING_ENABLED
void VmaBlockVector::PrintDetailedMap(class VmaJsonWriter& json)
{
    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);


    json.BeginObject();
    for (size_t i = 0; i < m_Blocks.size(); ++i)
    {
        json.BeginString();
        json.ContinueString(m_Blocks[i]->GetId());
        json.EndString();

        json.BeginObject();
        json.WriteString("MapRefCount");
        json.WriteNumber(m_Blocks[i]->GetMapRefCount());

        m_Blocks[i]->m_pMetadata->PrintDetailedMap(json);
        json.EndObject();
    }
    json.EndObject();
}
#endif // VMA_STATS_STRING_ENABLED

VkResult VmaBlockVector::CheckCorruption()
{
    if (!IsCorruptionDetectionEnabled())
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VmaMutexLockRead lock(m_Mutex, m_hAllocator->m_UseMutex);
    for (uint32_t blockIndex = 0; blockIndex < m_Blocks.size(); ++blockIndex)
    {
        VmaDeviceMemoryBlock* const pBlock = m_Blocks[blockIndex];
        VMA_ASSERT(pBlock);
        VkResult res = pBlock->CheckCorruption(m_hAllocator);
        if (res != VK_SUCCESS)
        {
            return res;
        }
    }
    return VK_SUCCESS;
}

#endif // _VMA_BLOCK_VECTOR_FUNCTIONS

#ifndef _VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS
VmaDefragmentationContext_T::VmaDefragmentationContext_T(
    VmaAllocator hAllocator,
    const VmaDefragmentationInfo& info)
    : m_MaxPassBytes(info.maxBytesPerPass == 0 ? VK_WHOLE_SIZE : info.maxBytesPerPass),
    m_MaxPassAllocations(info.maxAllocationsPerPass == 0 ? UINT32_MAX : info.maxAllocationsPerPass),
    m_MoveAllocator(hAllocator->GetAllocationCallbacks()),
    m_Moves(m_MoveAllocator)
{
    m_Algorithm = info.flags & VMA_DEFRAGMENTATION_FLAG_ALGORITHM_MASK;

    if (info.pool != VMA_NULL)
    {
        m_BlockVectorCount = 1;
        m_PoolBlockVector = &info.pool->m_BlockVector;
        m_pBlockVectors = &m_PoolBlockVector;
        m_PoolBlockVector->SetIncrementalSort(false);
        m_PoolBlockVector->SortByFreeSize();
    }
    else
    {
        m_BlockVectorCount = hAllocator->GetMemoryTypeCount();
        m_PoolBlockVector = VMA_NULL;
        m_pBlockVectors = hAllocator->m_pBlockVectors;
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            VmaBlockVector* vector = m_pBlockVectors[i];
            if (vector != VMA_NULL)
            {
                vector->SetIncrementalSort(false);
                vector->SortByFreeSize();
            }
        }
    }

    switch (m_Algorithm)
    {
    case 0: // Default algorithm
        m_Algorithm = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT;
        m_AlgorithmState = vma_new_array(hAllocator, StateBalanced, m_BlockVectorCount);
        break;
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        m_AlgorithmState = vma_new_array(hAllocator, StateBalanced, m_BlockVectorCount);
        break;
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        if (hAllocator->GetBufferImageGranularity() > 1)
        {
            m_AlgorithmState = vma_new_array(hAllocator, StateExtensive, m_BlockVectorCount);
        }
        break;
    }
}

VmaDefragmentationContext_T::~VmaDefragmentationContext_T()
{
    if (m_PoolBlockVector != VMA_NULL)
    {
        m_PoolBlockVector->SetIncrementalSort(true);
    }
    else
    {
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            VmaBlockVector* vector = m_pBlockVectors[i];
            if (vector != VMA_NULL)
                vector->SetIncrementalSort(true);
        }
    }

    if (m_AlgorithmState)
    {
        switch (m_Algorithm)
        {
        case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
            vma_delete_array(m_MoveAllocator.m_pCallbacks, reinterpret_cast<StateBalanced*>(m_AlgorithmState), m_BlockVectorCount);
            break;
        case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
            vma_delete_array(m_MoveAllocator.m_pCallbacks, reinterpret_cast<StateExtensive*>(m_AlgorithmState), m_BlockVectorCount);
            break;
        default:
            VMA_ASSERT(0);
        }
    }
}

VkResult VmaDefragmentationContext_T::DefragmentPassBegin(VmaDefragmentationPassMoveInfo& moveInfo)
{
    if (m_PoolBlockVector != VMA_NULL)
    {
        VmaMutexLockWrite lock(m_PoolBlockVector->GetMutex(), m_PoolBlockVector->GetAllocator()->m_UseMutex);

        if (m_PoolBlockVector->GetBlockCount() > 1)
            ComputeDefragmentation(*m_PoolBlockVector, 0);
        else if (m_PoolBlockVector->GetBlockCount() == 1)
            ReallocWithinBlock(*m_PoolBlockVector, m_PoolBlockVector->GetBlock(0));
    }
    else
    {
        for (uint32_t i = 0; i < m_BlockVectorCount; ++i)
        {
            if (m_pBlockVectors[i] != VMA_NULL)
            {
                VmaMutexLockWrite lock(m_pBlockVectors[i]->GetMutex(), m_pBlockVectors[i]->GetAllocator()->m_UseMutex);

                if (m_pBlockVectors[i]->GetBlockCount() > 1)
                {
                    if (ComputeDefragmentation(*m_pBlockVectors[i], i))
                        break;
                }
                else if (m_pBlockVectors[i]->GetBlockCount() == 1)
                {
                    if (ReallocWithinBlock(*m_pBlockVectors[i], m_pBlockVectors[i]->GetBlock(0)))
                        break;
                }
            }
        }
    }

    moveInfo.moveCount = static_cast<uint32_t>(m_Moves.size());
    if (moveInfo.moveCount > 0)
    {
        moveInfo.pMoves = m_Moves.data();
        return VK_INCOMPLETE;
    }

    moveInfo.pMoves = VMA_NULL;
    return VK_SUCCESS;
}

VkResult VmaDefragmentationContext_T::DefragmentPassEnd(VmaDefragmentationPassMoveInfo& moveInfo)
{
    VMA_ASSERT(moveInfo.moveCount > 0 ? moveInfo.pMoves != VMA_NULL : true);

    VkResult result = VK_SUCCESS;
    VmaStlAllocator<FragmentedBlock> blockAllocator(m_MoveAllocator.m_pCallbacks);
    VmaVector<FragmentedBlock, VmaStlAllocator<FragmentedBlock>> immovableBlocks(blockAllocator);
    VmaVector<FragmentedBlock, VmaStlAllocator<FragmentedBlock>> mappedBlocks(blockAllocator);

    VmaAllocator allocator = VMA_NULL;
    for (uint32_t i = 0; i < moveInfo.moveCount; ++i)
    {
        VmaDefragmentationMove& move = moveInfo.pMoves[i];
        size_t prevCount = 0, currentCount = 0;
        VkDeviceSize freedBlockSize = 0;

        uint32_t vectorIndex;
        VmaBlockVector* vector;
        if (m_PoolBlockVector != VMA_NULL)
        {
            vectorIndex = 0;
            vector = m_PoolBlockVector;
        }
        else
        {
            vectorIndex = move.srcAllocation->GetMemoryTypeIndex();
            vector = m_pBlockVectors[vectorIndex];
            VMA_ASSERT(vector != VMA_NULL);
        }

        switch (move.operation)
        {
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_COPY:
        {
            uint8_t mapCount = move.srcAllocation->SwapBlockAllocation(vector->m_hAllocator, move.dstTmpAllocation);
            if (mapCount > 0)
            {
                allocator = vector->m_hAllocator;
                VmaDeviceMemoryBlock* newMapBlock = move.srcAllocation->GetBlock();
                bool notPresent = true;
                for (FragmentedBlock& block : mappedBlocks)
                {
                    if (block.block == newMapBlock)
                    {
                        notPresent = false;
                        block.data += mapCount;
                        break;
                    }
                }
                if (notPresent)
                    mappedBlocks.push_back({ mapCount, newMapBlock });
            }

            // Scope for locks, Free have it's own lock
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                prevCount = vector->GetBlockCount();
                freedBlockSize = move.dstTmpAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.dstTmpAllocation);
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                currentCount = vector->GetBlockCount();
            }

            result = VK_INCOMPLETE;
            break;
        }
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE:
        {
            m_PassStats.bytesMoved -= move.srcAllocation->GetSize();
            --m_PassStats.allocationsMoved;
            vector->Free(move.dstTmpAllocation);

            VmaDeviceMemoryBlock* newBlock = move.srcAllocation->GetBlock();
            bool notPresent = true;
            for (const FragmentedBlock& block : immovableBlocks)
            {
                if (block.block == newBlock)
                {
                    notPresent = false;
                    break;
                }
            }
            if (notPresent)
                immovableBlocks.push_back({ vectorIndex, newBlock });
            break;
        }
        case VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY:
        {
            m_PassStats.bytesMoved -= move.srcAllocation->GetSize();
            --m_PassStats.allocationsMoved;
            // Scope for locks, Free have it's own lock
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                prevCount = vector->GetBlockCount();
                freedBlockSize = move.srcAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.srcAllocation);
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                currentCount = vector->GetBlockCount();
            }
            freedBlockSize *= prevCount - currentCount;

            VkDeviceSize dstBlockSize;
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                dstBlockSize = move.dstTmpAllocation->GetBlock()->m_pMetadata->GetSize();
            }
            vector->Free(move.dstTmpAllocation);
            {
                VmaMutexLockRead lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);
                freedBlockSize += dstBlockSize * (currentCount - vector->GetBlockCount());
                currentCount = vector->GetBlockCount();
            }

            result = VK_INCOMPLETE;
            break;
        }
        default:
            VMA_ASSERT(0);
        }

        if (prevCount > currentCount)
        {
            size_t freedBlocks = prevCount - currentCount;
            m_PassStats.deviceMemoryBlocksFreed += static_cast<uint32_t>(freedBlocks);
            m_PassStats.bytesFreed += freedBlockSize;
        }

        if(m_Algorithm == VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT &&
            m_AlgorithmState != VMA_NULL)
        {
            // Avoid unnecessary tries to allocate when new free block is available
            StateExtensive& state = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[vectorIndex];
            if (state.firstFreeBlock != SIZE_MAX)
            {
                const size_t diff = prevCount - currentCount;
                if (state.firstFreeBlock >= diff)
                {
                    state.firstFreeBlock -= diff;
                    if (state.firstFreeBlock != 0)
                        state.firstFreeBlock -= vector->GetBlock(state.firstFreeBlock - 1)->m_pMetadata->IsEmpty();
                }
                else
                    state.firstFreeBlock = 0;
            }
        }
    }
    moveInfo.moveCount = 0;
    moveInfo.pMoves = VMA_NULL;
    m_Moves.clear();

    // Update stats
    m_GlobalStats.allocationsMoved += m_PassStats.allocationsMoved;
    m_GlobalStats.bytesFreed += m_PassStats.bytesFreed;
    m_GlobalStats.bytesMoved += m_PassStats.bytesMoved;
    m_GlobalStats.deviceMemoryBlocksFreed += m_PassStats.deviceMemoryBlocksFreed;
    m_PassStats = { 0 };

    // Move blocks with immovable allocations according to algorithm
    if (immovableBlocks.size() > 0)
    {
        do
        {
            if(m_Algorithm == VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT)
            {
                if (m_AlgorithmState != VMA_NULL)
                {
                    bool swapped = false;
                    // Move to the start of free blocks range
                    for (const FragmentedBlock& block : immovableBlocks)
                    {
                        StateExtensive& state = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[block.data];
                        if (state.operation != StateExtensive::Operation::Cleanup)
                        {
                            VmaBlockVector* vector = m_pBlockVectors[block.data];
                            VmaMutexLockWrite lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);

                            for (size_t i = 0, count = vector->GetBlockCount() - m_ImmovableBlockCount; i < count; ++i)
                            {
                                if (vector->GetBlock(i) == block.block)
                                {
                                    VMA_SWAP(vector->m_Blocks[i], vector->m_Blocks[vector->GetBlockCount() - ++m_ImmovableBlockCount]);
                                    if (state.firstFreeBlock != SIZE_MAX)
                                    {
                                        if (i + 1 < state.firstFreeBlock)
                                        {
                                            if (state.firstFreeBlock > 1)
                                                VMA_SWAP(vector->m_Blocks[i], vector->m_Blocks[--state.firstFreeBlock]);
                                            else
                                                --state.firstFreeBlock;
                                        }
                                    }
                                    swapped = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (swapped)
                        result = VK_INCOMPLETE;
                    break;
                }
            }

            // Move to the beginning
            for (const FragmentedBlock& block : immovableBlocks)
            {
                VmaBlockVector* vector = m_pBlockVectors[block.data];
                VmaMutexLockWrite lock(vector->GetMutex(), vector->GetAllocator()->m_UseMutex);

                for (size_t i = m_ImmovableBlockCount; i < vector->GetBlockCount(); ++i)
                {
                    if (vector->GetBlock(i) == block.block)
                    {
                        VMA_SWAP(vector->m_Blocks[i], vector->m_Blocks[m_ImmovableBlockCount++]);
                        break;
                    }
                }
            }
        } while (false);
    }

    // Bulk-map destination blocks
    for (const FragmentedBlock& block : mappedBlocks)
    {
        VkResult res = block.block->Map(allocator, block.data, VMA_NULL);
        VMA_ASSERT(res == VK_SUCCESS);
    }
    return result;
}

bool VmaDefragmentationContext_T::ComputeDefragmentation(VmaBlockVector& vector, size_t index)
{
    switch (m_Algorithm)
    {
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT:
        return ComputeDefragmentation_Fast(vector);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_BALANCED_BIT:
        return ComputeDefragmentation_Balanced(vector, index, true);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FULL_BIT:
        return ComputeDefragmentation_Full(vector);
    case VMA_DEFRAGMENTATION_FLAG_ALGORITHM_EXTENSIVE_BIT:
        return ComputeDefragmentation_Extensive(vector, index);
    default:
        VMA_ASSERT(0);
        return ComputeDefragmentation_Balanced(vector, index, true);
    }
}

VmaDefragmentationContext_T::MoveAllocationData VmaDefragmentationContext_T::GetMoveData(
    VmaAllocHandle handle, VmaBlockMetadata* metadata)
{
    MoveAllocationData moveData;
    moveData.move.srcAllocation = (VmaAllocation)metadata->GetAllocationUserData(handle);
    moveData.size = moveData.move.srcAllocation->GetSize();
    moveData.alignment = moveData.move.srcAllocation->GetAlignment();
    moveData.type = moveData.move.srcAllocation->GetSuballocationType();
    moveData.flags = 0;

    if (moveData.move.srcAllocation->IsPersistentMap())
        moveData.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (moveData.move.srcAllocation->IsMappingAllowed())
        moveData.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    return moveData;
}

VmaDefragmentationContext_T::CounterStatus VmaDefragmentationContext_T::CheckCounters(VkDeviceSize bytes)
{
    // Ignore allocation if will exceed max size for copy
    if (m_PassStats.bytesMoved + bytes > m_MaxPassBytes)
    {
        if (++m_IgnoredAllocs < MAX_ALLOCS_TO_IGNORE)
            return CounterStatus::Ignore;
        else
            return CounterStatus::End;
    }
    else
        m_IgnoredAllocs = 0;
    return CounterStatus::Pass;
}

bool VmaDefragmentationContext_T::IncrementCounters(VkDeviceSize bytes)
{
    m_PassStats.bytesMoved += bytes;
    // Early return when max found
    if (++m_PassStats.allocationsMoved >= m_MaxPassAllocations || m_PassStats.bytesMoved >= m_MaxPassBytes)
    {
        VMA_ASSERT((m_PassStats.allocationsMoved == m_MaxPassAllocations ||
            m_PassStats.bytesMoved == m_MaxPassBytes) && "Exceeded maximal pass threshold!");
        return true;
    }
    return false;
}

bool VmaDefragmentationContext_T::ReallocWithinBlock(VmaBlockVector& vector, VmaDeviceMemoryBlock* block)
{
    VmaBlockMetadata* metadata = block->m_pMetadata;

    for (VmaAllocHandle handle = metadata->GetAllocationListBegin();
        handle != VK_NULL_HANDLE;
        handle = metadata->GetNextAllocation(handle))
    {
        MoveAllocationData moveData = GetMoveData(handle, metadata);
        // Ignore newly created allocations by defragmentation algorithm
        if (moveData.move.srcAllocation->GetUserData() == this)
            continue;
        switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
        {
        case CounterStatus::Ignore:
            continue;
        case CounterStatus::End:
            return true;
        case CounterStatus::Pass:
            break;
        default:
            VMA_ASSERT(0);
        }

        VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
        if (offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
        {
            VmaAllocationRequest request = {};
            if (metadata->CreateAllocationRequest(
                moveData.size,
                moveData.alignment,
                false,
                moveData.type,
                VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                &request))
            {
                if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                {
                    if (vector.CommitAllocationRequest(
                        request,
                        block,
                        moveData.alignment,
                        moveData.flags,
                        this,
                        moveData.type,
                        &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                    {
                        m_Moves.push_back(moveData.move);
                        if (IncrementCounters(moveData.size))
                            return true;
                    }
                }
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::AllocInOtherBlock(size_t start, size_t end, MoveAllocationData& data, VmaBlockVector& vector)
{
    for (; start < end; ++start)
    {
        VmaDeviceMemoryBlock* dstBlock = vector.GetBlock(start);
        if (dstBlock->m_pMetadata->GetSumFreeSize() >= data.size)
        {
            if (vector.AllocateFromBlock(dstBlock,
                data.size,
                data.alignment,
                data.flags,
                this,
                data.type,
                0,
                &data.move.dstTmpAllocation) == VK_SUCCESS)
            {
                m_Moves.push_back(data.move);
                if (IncrementCounters(data.size))
                    return true;
                break;
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::ComputeDefragmentation_Fast(VmaBlockVector& vector)
{
    // Move only between blocks

    // Go through allocations in last blocks and try to fit them inside first ones
    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        VmaBlockMetadata* metadata = vector.GetBlock(i)->m_pMetadata;

        for (VmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::ComputeDefragmentation_Balanced(VmaBlockVector& vector, size_t index, bool update)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0),
    // but only if there are noticeable gaps between them (some heuristic, ex. average size of allocation in block)
    VMA_ASSERT(m_AlgorithmState != VMA_NULL);

    StateBalanced& vectorState = reinterpret_cast<StateBalanced*>(m_AlgorithmState)[index];
    if (update && vectorState.avgAllocSize == UINT64_MAX)
        UpdateVectorStatistics(vector, vectorState);

    const size_t startMoveCount = m_Moves.size();
    VkDeviceSize minimalFreeRegion = vectorState.avgFreeSize / 2;
    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        VmaDeviceMemoryBlock* block = vector.GetBlock(i);
        VmaBlockMetadata* metadata = block->m_pMetadata;
        VkDeviceSize prevFreeRegionSize = 0;

        for (VmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = m_Moves.size();
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;

            VkDeviceSize nextFreeRegionSize = metadata->GetNextFreeRegionSize(handle);
            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
            if (prevMoveCount == m_Moves.size() && offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
            {
                // Check if realloc will make sense
                if (prevFreeRegionSize >= minimalFreeRegion ||
                    nextFreeRegionSize >= minimalFreeRegion ||
                    moveData.size <= vectorState.avgFreeSize ||
                    moveData.size <= vectorState.avgAllocSize)
                {
                    VmaAllocationRequest request = {};
                    if (metadata->CreateAllocationRequest(
                        moveData.size,
                        moveData.alignment,
                        false,
                        moveData.type,
                        VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                        &request))
                    {
                        if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                        {
                            if (vector.CommitAllocationRequest(
                                request,
                                block,
                                moveData.alignment,
                                moveData.flags,
                                this,
                                moveData.type,
                                &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                            {
                                m_Moves.push_back(moveData.move);
                                if (IncrementCounters(moveData.size))
                                    return true;
                            }
                        }
                    }
                }
            }
            prevFreeRegionSize = nextFreeRegionSize;
        }
    }

    // No moves performed, update statistics to current vector state
    if (startMoveCount == m_Moves.size() && !update)
    {
        vectorState.avgAllocSize = UINT64_MAX;
        return ComputeDefragmentation_Balanced(vector, index, false);
    }
    return false;
}

bool VmaDefragmentationContext_T::ComputeDefragmentation_Full(VmaBlockVector& vector)
{
    // Go over every allocation and try to fit it in previous blocks at lowest offsets,
    // if not possible: realloc within single block to minimize offset (exclude offset == 0)

    for (size_t i = vector.GetBlockCount() - 1; i > m_ImmovableBlockCount; --i)
    {
        VmaDeviceMemoryBlock* block = vector.GetBlock(i);
        VmaBlockMetadata* metadata = block->m_pMetadata;

        for (VmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            const size_t prevMoveCount = m_Moves.size();
            if (AllocInOtherBlock(0, i, moveData, vector))
                return true;

            // If no room found then realloc within block for lower offset
            VkDeviceSize offset = moveData.move.srcAllocation->GetOffset();
            if (prevMoveCount == m_Moves.size() && offset != 0 && metadata->GetSumFreeSize() >= moveData.size)
            {
                VmaAllocationRequest request = {};
                if (metadata->CreateAllocationRequest(
                    moveData.size,
                    moveData.alignment,
                    false,
                    moveData.type,
                    VMA_ALLOCATION_CREATE_STRATEGY_MIN_OFFSET_BIT,
                    &request))
                {
                    if (metadata->GetAllocationOffset(request.allocHandle) < offset)
                    {
                        if (vector.CommitAllocationRequest(
                            request,
                            block,
                            moveData.alignment,
                            moveData.flags,
                            this,
                            moveData.type,
                            &moveData.move.dstTmpAllocation) == VK_SUCCESS)
                        {
                            m_Moves.push_back(moveData.move);
                            if (IncrementCounters(moveData.size))
                                return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool VmaDefragmentationContext_T::ComputeDefragmentation_Extensive(VmaBlockVector& vector, size_t index)
{
    // First free single block, then populate it to the brim, then free another block, and so on

    // Fallback to previous algorithm since without granularity conflicts it can achieve max packing
    if (vector.m_BufferImageGranularity == 1)
        return ComputeDefragmentation_Full(vector);

    VMA_ASSERT(m_AlgorithmState != VMA_NULL);

    StateExtensive& vectorState = reinterpret_cast<StateExtensive*>(m_AlgorithmState)[index];

    bool texturePresent = false, bufferPresent = false, otherPresent = false;
    switch (vectorState.operation)
    {
    case StateExtensive::Operation::Done: // Vector defragmented
        return false;
    case StateExtensive::Operation::FindFreeBlockBuffer:
    case StateExtensive::Operation::FindFreeBlockTexture:
    case StateExtensive::Operation::FindFreeBlockAll:
    {
        // No more blocks to free, just perform fast realloc and move to cleanup
        if (vectorState.firstFreeBlock == 0)
        {
            vectorState.operation = StateExtensive::Operation::Cleanup;
            return ComputeDefragmentation_Fast(vector);
        }

        // No free blocks, have to clear last one
        size_t last = (vectorState.firstFreeBlock == SIZE_MAX ? vector.GetBlockCount() : vectorState.firstFreeBlock) - 1;
        VmaBlockMetadata* freeMetadata = vector.GetBlock(last)->m_pMetadata;

        const size_t prevMoveCount = m_Moves.size();
        for (VmaAllocHandle handle = freeMetadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = freeMetadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, freeMetadata);
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Check all previous blocks for free space
            if (AllocInOtherBlock(0, last, moveData, vector))
            {
                // Full clear performed already
                if (prevMoveCount != m_Moves.size() && freeMetadata->GetNextAllocation(handle) == VK_NULL_HANDLE)
                    vectorState.firstFreeBlock = last;
                return true;
            }
        }

        if (prevMoveCount == m_Moves.size())
        {
            // Cannot perform full clear, have to move data in other blocks around
            if (last != 0)
            {
                for (size_t i = last - 1; i; --i)
                {
                    if (ReallocWithinBlock(vector, vector.GetBlock(i)))
                        return true;
                }
            }

            if (prevMoveCount == m_Moves.size())
            {
                // No possible reallocs within blocks, try to move them around fast
                return ComputeDefragmentation_Fast(vector);
            }
        }
        else
        {
            switch (vectorState.operation)
            {
            case StateExtensive::Operation::FindFreeBlockBuffer:
                vectorState.operation = StateExtensive::Operation::MoveBuffers;
                break;
            case StateExtensive::Operation::FindFreeBlockTexture:
                vectorState.operation = StateExtensive::Operation::MoveTextures;
                break;
            case StateExtensive::Operation::FindFreeBlockAll:
                vectorState.operation = StateExtensive::Operation::MoveAll;
                break;
            default:
                VMA_ASSERT(0);
                vectorState.operation = StateExtensive::Operation::MoveTextures;
            }
            vectorState.firstFreeBlock = last;
            // Nothing done, block found without reallocations, can perform another reallocs in same pass
            return ComputeDefragmentation_Extensive(vector, index);
        }
        break;
    }
    case StateExtensive::Operation::MoveTextures:
    {
        if (MoveDataToFreeBlocks(VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (texturePresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockTexture;
                return ComputeDefragmentation_Extensive(vector, index);
            }

            if (!bufferPresent && !otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more textures to move, check buffers
            vectorState.operation = StateExtensive::Operation::MoveBuffers;
            bufferPresent = false;
            otherPresent = false;
        }
        else
            break;
        VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveBuffers:
    {
        if (MoveDataToFreeBlocks(VMA_SUBALLOCATION_TYPE_BUFFER, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (bufferPresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockBuffer;
                return ComputeDefragmentation_Extensive(vector, index);
            }

            if (!otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::Cleanup;
                break;
            }

            // No more buffers to move, check all others
            vectorState.operation = StateExtensive::Operation::MoveAll;
            otherPresent = false;
        }
        else
            break;
        VMA_FALLTHROUGH; // Fallthrough
    }
    case StateExtensive::Operation::MoveAll:
    {
        if (MoveDataToFreeBlocks(VMA_SUBALLOCATION_TYPE_FREE, vector,
            vectorState.firstFreeBlock, texturePresent, bufferPresent, otherPresent))
        {
            if (otherPresent)
            {
                vectorState.operation = StateExtensive::Operation::FindFreeBlockBuffer;
                return ComputeDefragmentation_Extensive(vector, index);
            }
            // Everything moved
            vectorState.operation = StateExtensive::Operation::Cleanup;
        }
        break;
    }
    case StateExtensive::Operation::Cleanup:
        // Cleanup is handled below so that other operations may reuse the cleanup code. This case is here to prevent the unhandled enum value warning (C4062).
        break;
    }

    if (vectorState.operation == StateExtensive::Operation::Cleanup)
    {
        // All other work done, pack data in blocks even tighter if possible
        const size_t prevMoveCount = m_Moves.size();
        for (size_t i = 0; i < vector.GetBlockCount(); ++i)
        {
            if (ReallocWithinBlock(vector, vector.GetBlock(i)))
                return true;
        }

        if (prevMoveCount == m_Moves.size())
            vectorState.operation = StateExtensive::Operation::Done;
    }
    return false;
}

void VmaDefragmentationContext_T::UpdateVectorStatistics(VmaBlockVector& vector, StateBalanced& state)
{
    size_t allocCount = 0;
    size_t freeCount = 0;
    state.avgFreeSize = 0;
    state.avgAllocSize = 0;

    for (size_t i = 0; i < vector.GetBlockCount(); ++i)
    {
        VmaBlockMetadata* metadata = vector.GetBlock(i)->m_pMetadata;

        allocCount += metadata->GetAllocationCount();
        freeCount += metadata->GetFreeRegionsCount();
        state.avgFreeSize += metadata->GetSumFreeSize();
        state.avgAllocSize += metadata->GetSize();
    }

    state.avgAllocSize = (state.avgAllocSize - state.avgFreeSize) / allocCount;
    state.avgFreeSize /= freeCount;
}

bool VmaDefragmentationContext_T::MoveDataToFreeBlocks(VmaSuballocationType currentType,
    VmaBlockVector& vector, size_t firstFreeBlock,
    bool& texturePresent, bool& bufferPresent, bool& otherPresent)
{
    const size_t prevMoveCount = m_Moves.size();
    for (size_t i = firstFreeBlock ; i;)
    {
        VmaDeviceMemoryBlock* block = vector.GetBlock(--i);
        VmaBlockMetadata* metadata = block->m_pMetadata;

        for (VmaAllocHandle handle = metadata->GetAllocationListBegin();
            handle != VK_NULL_HANDLE;
            handle = metadata->GetNextAllocation(handle))
        {
            MoveAllocationData moveData = GetMoveData(handle, metadata);
            // Ignore newly created allocations by defragmentation algorithm
            if (moveData.move.srcAllocation->GetUserData() == this)
                continue;
            switch (CheckCounters(moveData.move.srcAllocation->GetSize()))
            {
            case CounterStatus::Ignore:
                continue;
            case CounterStatus::End:
                return true;
            case CounterStatus::Pass:
                break;
            default:
                VMA_ASSERT(0);
            }

            // Move only single type of resources at once
            if (!VmaIsBufferImageGranularityConflict(moveData.type, currentType))
            {
                // Try to fit allocation into free blocks
                if (AllocInOtherBlock(firstFreeBlock, vector.GetBlockCount(), moveData, vector))
                    return false;
            }

            if (!VmaIsBufferImageGranularityConflict(moveData.type, VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL))
                texturePresent = true;
            else if (!VmaIsBufferImageGranularityConflict(moveData.type, VMA_SUBALLOCATION_TYPE_BUFFER))
                bufferPresent = true;
            else
                otherPresent = true;
        }
    }
    return prevMoveCount == m_Moves.size();
}
#endif // _VMA_DEFRAGMENTATION_CONTEXT_FUNCTIONS

#ifndef _VMA_POOL_T_FUNCTIONS
VmaPool_T::VmaPool_T(
    VmaAllocator hAllocator,
    const VmaPoolCreateInfo& createInfo,
    VkDeviceSize preferredBlockSize)
    : m_BlockVector(
        hAllocator,
        this, // hParentPool
        createInfo.memoryTypeIndex,
        createInfo.blockSize != 0 ? createInfo.blockSize : preferredBlockSize,
        createInfo.minBlockCount,
        createInfo.maxBlockCount,
        (createInfo.flags& VMA_POOL_CREATE_IGNORE_BUFFER_IMAGE_GRANULARITY_BIT) != 0 ? 1 : hAllocator->GetBufferImageGranularity(),
        createInfo.blockSize != 0, // explicitBlockSize
        createInfo.flags & VMA_POOL_CREATE_ALGORITHM_MASK, // algorithm
        createInfo.priority,
        VMA_MAX(hAllocator->GetMemoryTypeMinAlignment(createInfo.memoryTypeIndex), createInfo.minAllocationAlignment),
        createInfo.pMemoryAllocateNext),
    m_Id(0),
    m_Name(VMA_NULL) {}

VmaPool_T::~VmaPool_T()
{
    VMA_ASSERT(m_PrevPool == VMA_NULL && m_NextPool == VMA_NULL);
}

void VmaPool_T::SetName(const char* pName)
{
    const VkAllocationCallbacks* allocs = m_BlockVector.GetAllocator()->GetAllocationCallbacks();
    VmaFreeString(allocs, m_Name);

    if (pName != VMA_NULL)
    {
        m_Name = VmaCreateStringCopy(allocs, pName);
    }
    else
    {
        m_Name = VMA_NULL;
    }
}
#endif // _VMA_POOL_T_FUNCTIONS

#ifndef _VMA_ALLOCATOR_T_FUNCTIONS
VmaAllocator_T::VmaAllocator_T(const VmaAllocatorCreateInfo* pCreateInfo) :
    m_UseMutex((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT) == 0),
    m_VulkanApiVersion(pCreateInfo->vulkanApiVersion != 0 ? pCreateInfo->vulkanApiVersion : VK_API_VERSION_1_0),
    m_UseKhrDedicatedAllocation((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0),
    m_UseKhrBindMemory2((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0),
    m_UseExtMemoryBudget((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0),
    m_UseAmdDeviceCoherentMemory((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT) != 0),
    m_UseKhrBufferDeviceAddress((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) != 0),
    m_UseExtMemoryPriority((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT) != 0),
    m_hDevice(pCreateInfo->device),
    m_hInstance(pCreateInfo->instance),
    m_AllocationCallbacksSpecified(pCreateInfo->pAllocationCallbacks != VMA_NULL),
    m_AllocationCallbacks(pCreateInfo->pAllocationCallbacks ?
        *pCreateInfo->pAllocationCallbacks : VmaEmptyAllocationCallbacks),
    m_AllocationObjectAllocator(&m_AllocationCallbacks),
    m_HeapSizeLimitMask(0),
    m_DeviceMemoryCount(0),
    m_PreferredLargeHeapBlockSize(0),
    m_PhysicalDevice(pCreateInfo->physicalDevice),
    m_GpuDefragmentationMemoryTypeBits(UINT32_MAX),
    m_NextPoolId(0),
    m_GlobalMemoryTypeBits(UINT32_MAX)
{
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_UseKhrDedicatedAllocation = false;
        m_UseKhrBindMemory2 = false;
    }

    if(VMA_DEBUG_DETECT_CORRUPTION)
    {
        // Needs to be multiply of uint32_t size because we are going to write VMA_CORRUPTION_DETECTION_MAGIC_VALUE to it.
        VMA_ASSERT(VMA_DEBUG_MARGIN % sizeof(uint32_t) == 0);
    }

    VMA_ASSERT(pCreateInfo->physicalDevice && pCreateInfo->device && pCreateInfo->instance);

    if(m_VulkanApiVersion < VK_MAKE_VERSION(1, 1, 0))
    {
#if !(VMA_DEDICATED_ALLOCATION)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT set but required extensions are disabled by preprocessor macros.");
        }
#endif
#if !(VMA_BIND_MEMORY2)
        if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT) != 0)
        {
            VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT set but required extension is disabled by preprocessor macros.");
        }
#endif
    }
#if !(VMA_MEMORY_BUDGET)
    if((pCreateInfo->flags & VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT) != 0)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set but required extension is disabled by preprocessor macros.");
    }
#endif
#if !(VMA_BUFFER_DEVICE_ADDRESS)
    if(m_UseKhrBufferDeviceAddress)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT is set but required extension or Vulkan 1.2 is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif
#if VMA_VULKAN_VERSION < 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        VMA_ASSERT(0 && "vulkanApiVersion >= VK_API_VERSION_1_3 but required Vulkan version is disabled by preprocessor macros.");
    }
#endif
#if VMA_VULKAN_VERSION < 1002000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 2, 0))
    {
        VMA_ASSERT(0 && "vulkanApiVersion >= VK_API_VERSION_1_2 but required Vulkan version is disabled by preprocessor macros.");
    }
#endif
#if VMA_VULKAN_VERSION < 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(0 && "vulkanApiVersion >= VK_API_VERSION_1_1 but required Vulkan version is disabled by preprocessor macros.");
    }
#endif
#if !(VMA_MEMORY_PRIORITY)
    if(m_UseExtMemoryPriority)
    {
        VMA_ASSERT(0 && "VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT is set but required extension is not available in your Vulkan header or its support in VMA has been disabled by a preprocessor macro.");
    }
#endif

    memset(&m_DeviceMemoryCallbacks, 0 ,sizeof(m_DeviceMemoryCallbacks));
    memset(&m_PhysicalDeviceProperties, 0, sizeof(m_PhysicalDeviceProperties));
    memset(&m_MemProps, 0, sizeof(m_MemProps));

    memset(&m_pBlockVectors, 0, sizeof(m_pBlockVectors));
    memset(&m_VulkanFunctions, 0, sizeof(m_VulkanFunctions));

#if VMA_EXTERNAL_MEMORY
    memset(&m_TypeExternalMemoryHandleTypes, 0, sizeof(m_TypeExternalMemoryHandleTypes));
#endif // #if VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pDeviceMemoryCallbacks != VMA_NULL)
    {
        m_DeviceMemoryCallbacks.pUserData = pCreateInfo->pDeviceMemoryCallbacks->pUserData;
        m_DeviceMemoryCallbacks.pfnAllocate = pCreateInfo->pDeviceMemoryCallbacks->pfnAllocate;
        m_DeviceMemoryCallbacks.pfnFree = pCreateInfo->pDeviceMemoryCallbacks->pfnFree;
    }

    ImportVulkanFunctions(pCreateInfo->pVulkanFunctions);

    (*m_VulkanFunctions.vkGetPhysicalDeviceProperties)(m_PhysicalDevice, &m_PhysicalDeviceProperties);
    (*m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties)(m_PhysicalDevice, &m_MemProps);

    VMA_ASSERT(VmaIsPow2(VMA_MIN_ALIGNMENT));
    VMA_ASSERT(VmaIsPow2(VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY));
    VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.bufferImageGranularity));
    VMA_ASSERT(VmaIsPow2(m_PhysicalDeviceProperties.limits.nonCoherentAtomSize));

    m_PreferredLargeHeapBlockSize = (pCreateInfo->preferredLargeHeapBlockSize != 0) ?
        pCreateInfo->preferredLargeHeapBlockSize : static_cast<VkDeviceSize>(VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE);

    m_GlobalMemoryTypeBits = CalculateGlobalMemoryTypeBits();

#if VMA_EXTERNAL_MEMORY
    if(pCreateInfo->pTypeExternalMemoryHandleTypes != VMA_NULL)
    {
        memcpy(m_TypeExternalMemoryHandleTypes, pCreateInfo->pTypeExternalMemoryHandleTypes,
            sizeof(VkExternalMemoryHandleTypeFlagsKHR) * GetMemoryTypeCount());
    }
#endif // #if VMA_EXTERNAL_MEMORY

    if(pCreateInfo->pHeapSizeLimit != VMA_NULL)
    {
        for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
        {
            const VkDeviceSize limit = pCreateInfo->pHeapSizeLimit[heapIndex];
            if(limit != VK_WHOLE_SIZE)
            {
                m_HeapSizeLimitMask |= 1u << heapIndex;
                if(limit < m_MemProps.memoryHeaps[heapIndex].size)
                {
                    m_MemProps.memoryHeaps[heapIndex].size = limit;
                }
            }
        }
    }

    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        // Create only supported types
        if((m_GlobalMemoryTypeBits & (1u << memTypeIndex)) != 0)
        {
            const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(memTypeIndex);
            m_pBlockVectors[memTypeIndex] = vma_new(this, VmaBlockVector)(
                this,
                VK_NULL_HANDLE, // hParentPool
                memTypeIndex,
                preferredBlockSize,
                0,
                SIZE_MAX,
                GetBufferImageGranularity(),
                false, // explicitBlockSize
                0, // algorithm
                0.5f, // priority (0.5 is the default per Vulkan spec)
                GetMemoryTypeMinAlignment(memTypeIndex), // minAllocationAlignment
                VMA_NULL); // // pMemoryAllocateNext
            // No need to call m_pBlockVectors[memTypeIndex][blockVectorTypeIndex]->CreateMinBlocks here,
            // becase minBlockCount is 0.
        }
    }
}

VkResult VmaAllocator_T::Init(const VmaAllocatorCreateInfo* pCreateInfo)
{
    VkResult res = VK_SUCCESS;

#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if VMA_MEMORY_BUDGET

    return res;
}

VmaAllocator_T::~VmaAllocator_T()
{
    VMA_ASSERT(m_Pools.IsEmpty());

    for(size_t memTypeIndex = GetMemoryTypeCount(); memTypeIndex--; )
    {
        vma_delete(this, m_pBlockVectors[memTypeIndex]);
    }
}

void VmaAllocator_T::ImportVulkanFunctions(const VmaVulkanFunctions* pVulkanFunctions)
{
#if VMA_STATIC_VULKAN_FUNCTIONS == 1
    ImportVulkanFunctions_Static();
#endif

    if(pVulkanFunctions != VMA_NULL)
    {
        ImportVulkanFunctions_Custom(pVulkanFunctions);
    }

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1
    ImportVulkanFunctions_Dynamic();
#endif

    ValidateVulkanFunctions();
}

#if VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Static()
{
    // Vulkan 1.0
    m_VulkanFunctions.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr;
    m_VulkanFunctions.vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetDeviceProcAddr;
    m_VulkanFunctions.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
    m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetPhysicalDeviceMemoryProperties;
    m_VulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
    m_VulkanFunctions.vkFreeMemory = (PFN_vkFreeMemory)vkFreeMemory;
    m_VulkanFunctions.vkMapMemory = (PFN_vkMapMemory)vkMapMemory;
    m_VulkanFunctions.vkUnmapMemory = (PFN_vkUnmapMemory)vkUnmapMemory;
    m_VulkanFunctions.vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
    m_VulkanFunctions.vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
    m_VulkanFunctions.vkBindBufferMemory = (PFN_vkBindBufferMemory)vkBindBufferMemory;
    m_VulkanFunctions.vkBindImageMemory = (PFN_vkBindImageMemory)vkBindImageMemory;
    m_VulkanFunctions.vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
    m_VulkanFunctions.vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
    m_VulkanFunctions.vkCreateBuffer = (PFN_vkCreateBuffer)vkCreateBuffer;
    m_VulkanFunctions.vkDestroyBuffer = (PFN_vkDestroyBuffer)vkDestroyBuffer;
    m_VulkanFunctions.vkCreateImage = (PFN_vkCreateImage)vkCreateImage;
    m_VulkanFunctions.vkDestroyImage = (PFN_vkDestroyImage)vkDestroyImage;
    m_VulkanFunctions.vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;

    // Vulkan 1.1
#if VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
        m_VulkanFunctions.vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
        m_VulkanFunctions.vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
        m_VulkanFunctions.vkBindImageMemory2KHR = (PFN_vkBindImageMemory2)vkBindImageMemory2;
    }
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetPhysicalDeviceMemoryProperties2;
    }
#endif

#if VMA_VULKAN_VERSION >= 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements = (PFN_vkGetDeviceBufferMemoryRequirements)vkGetDeviceBufferMemoryRequirements;
        m_VulkanFunctions.vkGetDeviceImageMemoryRequirements = (PFN_vkGetDeviceImageMemoryRequirements)vkGetDeviceImageMemoryRequirements;
    }
#endif
}

#endif // VMA_STATIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Custom(const VmaVulkanFunctions* pVulkanFunctions)
{
    VMA_ASSERT(pVulkanFunctions != VMA_NULL);

#define VMA_COPY_IF_NOT_NULL(funcName) \
    if(pVulkanFunctions->funcName != VMA_NULL) m_VulkanFunctions.funcName = pVulkanFunctions->funcName;

    VMA_COPY_IF_NOT_NULL(vkGetInstanceProcAddr);
    VMA_COPY_IF_NOT_NULL(vkGetDeviceProcAddr);
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceProperties);
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties);
    VMA_COPY_IF_NOT_NULL(vkAllocateMemory);
    VMA_COPY_IF_NOT_NULL(vkFreeMemory);
    VMA_COPY_IF_NOT_NULL(vkMapMemory);
    VMA_COPY_IF_NOT_NULL(vkUnmapMemory);
    VMA_COPY_IF_NOT_NULL(vkFlushMappedMemoryRanges);
    VMA_COPY_IF_NOT_NULL(vkInvalidateMappedMemoryRanges);
    VMA_COPY_IF_NOT_NULL(vkBindBufferMemory);
    VMA_COPY_IF_NOT_NULL(vkBindImageMemory);
    VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkCreateBuffer);
    VMA_COPY_IF_NOT_NULL(vkDestroyBuffer);
    VMA_COPY_IF_NOT_NULL(vkCreateImage);
    VMA_COPY_IF_NOT_NULL(vkDestroyImage);
    VMA_COPY_IF_NOT_NULL(vkCmdCopyBuffer);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkGetBufferMemoryRequirements2KHR);
    VMA_COPY_IF_NOT_NULL(vkGetImageMemoryRequirements2KHR);
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkBindBufferMemory2KHR);
    VMA_COPY_IF_NOT_NULL(vkBindImageMemory2KHR);
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    VMA_COPY_IF_NOT_NULL(vkGetPhysicalDeviceMemoryProperties2KHR);
#endif

#if VMA_VULKAN_VERSION >= 1003000
    VMA_COPY_IF_NOT_NULL(vkGetDeviceBufferMemoryRequirements);
    VMA_COPY_IF_NOT_NULL(vkGetDeviceImageMemoryRequirements);
#endif

#undef VMA_COPY_IF_NOT_NULL
}

#if VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ImportVulkanFunctions_Dynamic()
{
    VMA_ASSERT(m_VulkanFunctions.vkGetInstanceProcAddr && m_VulkanFunctions.vkGetDeviceProcAddr &&
        "To use VMA_DYNAMIC_VULKAN_FUNCTIONS in new versions of VMA you now have to pass "
        "VmaVulkanFunctions::vkGetInstanceProcAddr and vkGetDeviceProcAddr as VmaAllocatorCreateInfo::pVulkanFunctions. "
        "Other members can be null.");

#define VMA_FETCH_INSTANCE_FUNC(memberName, functionPointerType, functionNameString) \
    if(m_VulkanFunctions.memberName == VMA_NULL) \
        m_VulkanFunctions.memberName = \
            (functionPointerType)m_VulkanFunctions.vkGetInstanceProcAddr(m_hInstance, functionNameString);
#define VMA_FETCH_DEVICE_FUNC(memberName, functionPointerType, functionNameString) \
    if(m_VulkanFunctions.memberName == VMA_NULL) \
        m_VulkanFunctions.memberName = \
            (functionPointerType)m_VulkanFunctions.vkGetDeviceProcAddr(m_hDevice, functionNameString);

    VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceProperties, PFN_vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
    VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties, PFN_vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
    VMA_FETCH_DEVICE_FUNC(vkAllocateMemory, PFN_vkAllocateMemory, "vkAllocateMemory");
    VMA_FETCH_DEVICE_FUNC(vkFreeMemory, PFN_vkFreeMemory, "vkFreeMemory");
    VMA_FETCH_DEVICE_FUNC(vkMapMemory, PFN_vkMapMemory, "vkMapMemory");
    VMA_FETCH_DEVICE_FUNC(vkUnmapMemory, PFN_vkUnmapMemory, "vkUnmapMemory");
    VMA_FETCH_DEVICE_FUNC(vkFlushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
    VMA_FETCH_DEVICE_FUNC(vkInvalidateMappedMemoryRanges, PFN_vkInvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges");
    VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
    VMA_FETCH_DEVICE_FUNC(vkBindImageMemory, PFN_vkBindImageMemory, "vkBindImageMemory");
    VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements, PFN_vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
    VMA_FETCH_DEVICE_FUNC(vkCreateBuffer, PFN_vkCreateBuffer, "vkCreateBuffer");
    VMA_FETCH_DEVICE_FUNC(vkDestroyBuffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
    VMA_FETCH_DEVICE_FUNC(vkCreateImage, PFN_vkCreateImage, "vkCreateImage");
    VMA_FETCH_DEVICE_FUNC(vkDestroyImage, PFN_vkDestroyImage, "vkDestroyImage");
    VMA_FETCH_DEVICE_FUNC(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");

#if VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2, "vkGetBufferMemoryRequirements2");
        VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2, "vkGetImageMemoryRequirements2");
        VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2, "vkBindBufferMemory2");
        VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2, "vkBindImageMemory2");
    }
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2, "vkGetPhysicalDeviceMemoryProperties2");
    }
    else if(m_UseExtMemoryBudget)
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
#endif

#if VMA_DEDICATED_ALLOCATION
    if(m_UseKhrDedicatedAllocation)
    {
        VMA_FETCH_DEVICE_FUNC(vkGetBufferMemoryRequirements2KHR, PFN_vkGetBufferMemoryRequirements2KHR, "vkGetBufferMemoryRequirements2KHR");
        VMA_FETCH_DEVICE_FUNC(vkGetImageMemoryRequirements2KHR, PFN_vkGetImageMemoryRequirements2KHR, "vkGetImageMemoryRequirements2KHR");
    }
#endif

#if VMA_BIND_MEMORY2
    if(m_UseKhrBindMemory2)
    {
        VMA_FETCH_DEVICE_FUNC(vkBindBufferMemory2KHR, PFN_vkBindBufferMemory2KHR, "vkBindBufferMemory2KHR");
        VMA_FETCH_DEVICE_FUNC(vkBindImageMemory2KHR, PFN_vkBindImageMemory2KHR, "vkBindImageMemory2KHR");
    }
#endif // #if VMA_BIND_MEMORY2

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2");
    }
    else if(m_UseExtMemoryBudget)
    {
        VMA_FETCH_INSTANCE_FUNC(vkGetPhysicalDeviceMemoryProperties2KHR, PFN_vkGetPhysicalDeviceMemoryProperties2KHR, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }
#endif // #if VMA_MEMORY_BUDGET

#if VMA_VULKAN_VERSION >= 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceBufferMemoryRequirements, PFN_vkGetDeviceBufferMemoryRequirements, "vkGetDeviceBufferMemoryRequirements");
        VMA_FETCH_DEVICE_FUNC(vkGetDeviceImageMemoryRequirements, PFN_vkGetDeviceImageMemoryRequirements, "vkGetDeviceImageMemoryRequirements");
    }
#endif

#undef VMA_FETCH_DEVICE_FUNC
#undef VMA_FETCH_INSTANCE_FUNC
}

#endif // VMA_DYNAMIC_VULKAN_FUNCTIONS == 1

void VmaAllocator_T::ValidateVulkanFunctions()
{
    VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceProperties != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkAllocateMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkFreeMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkMapMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkUnmapMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkFlushMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkInvalidateMappedMemoryRanges != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCreateBuffer != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkDestroyBuffer != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCreateImage != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkDestroyImage != VMA_NULL);
    VMA_ASSERT(m_VulkanFunctions.vkCmdCopyBuffer != VMA_NULL);

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrDedicatedAllocation)
    {
        VMA_ASSERT(m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR != VMA_NULL);
        VMA_ASSERT(m_VulkanFunctions.vkGetImageMemoryRequirements2KHR != VMA_NULL);
    }
#endif

#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0) || m_UseKhrBindMemory2)
    {
        VMA_ASSERT(m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL);
        VMA_ASSERT(m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL);
    }
#endif

#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
    if(m_UseExtMemoryBudget || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VMA_ASSERT(m_VulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR != VMA_NULL);
    }
#endif

#if VMA_VULKAN_VERSION >= 1003000
    if(m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0))
    {
        VMA_ASSERT(m_VulkanFunctions.vkGetDeviceBufferMemoryRequirements != VMA_NULL);
        VMA_ASSERT(m_VulkanFunctions.vkGetDeviceImageMemoryRequirements != VMA_NULL);
    }
#endif
}

VkDeviceSize VmaAllocator_T::CalcPreferredBlockSize(uint32_t memTypeIndex)
{
    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
    const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
    const bool isSmallHeap = heapSize <= VMA_SMALL_HEAP_MAX_SIZE;
    return VmaAlignUp(isSmallHeap ? (heapSize / 8) : m_PreferredLargeHeapBlockSize, (VkDeviceSize)32);
}

VkResult VmaAllocator_T::AllocateMemoryOfType(
    VmaPool pool,
    VkDeviceSize size,
    VkDeviceSize alignment,
    bool dedicatedPreferred,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VkFlags dedicatedBufferImageUsage,
    const VmaAllocationCreateInfo& createInfo,
    uint32_t memTypeIndex,
    VmaSuballocationType suballocType,
    VmaDedicatedAllocationList& dedicatedAllocations,
    VmaBlockVector& blockVector,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations != VMA_NULL);
    VMA_DEBUG_LOG_FORMAT("  AllocateMemory: MemoryTypeIndex=%u, AllocationCount=%zu, Size=%llu", memTypeIndex, allocationCount, size);

    VmaAllocationCreateInfo finalCreateInfo = createInfo;
    VkResult res = CalcMemTypeParams(
        finalCreateInfo,
        memTypeIndex,
        size,
        allocationCount);
    if(res != VK_SUCCESS)
        return res;

    if((finalCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
    {
        return AllocateDedicatedMemory(
            pool,
            size,
            suballocType,
            dedicatedAllocations,
            memTypeIndex,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
            (finalCreateInfo.flags &
                (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
            finalCreateInfo.pUserData,
            finalCreateInfo.priority,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            allocationCount,
            pAllocations,
            blockVector.GetAllocationNextPtr());
    }
    else
    {
        const bool canAllocateDedicated =
            (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) == 0 &&
            (pool == VK_NULL_HANDLE || !blockVector.HasExplicitBlockSize());

        if(canAllocateDedicated)
        {
            // Heuristics: Allocate dedicated memory if requested size if greater than half of preferred block size.
            if(size > blockVector.GetPreferredBlockSize() / 2)
            {
                dedicatedPreferred = true;
            }
            // Protection against creating each allocation as dedicated when we reach or exceed heap size/budget,
            // which can quickly deplete maxMemoryAllocationCount: Don't prefer dedicated allocations when above
            // 3/4 of the maximum allocation count.
            if(m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount < UINT32_MAX / 4 &&
                m_DeviceMemoryCount.load() > m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount * 3 / 4)
            {
                dedicatedPreferred = false;
            }

            if(dedicatedPreferred)
            {
                res = AllocateDedicatedMemory(
                    pool,
                    size,
                    suballocType,
                    dedicatedAllocations,
                    memTypeIndex,
                    (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                    (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                    (finalCreateInfo.flags &
                        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
                    (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
                    finalCreateInfo.pUserData,
                    finalCreateInfo.priority,
                    dedicatedBuffer,
                    dedicatedImage,
                    dedicatedBufferImageUsage,
                    allocationCount,
                    pAllocations,
                    blockVector.GetAllocationNextPtr());
                if(res == VK_SUCCESS)
                {
                    // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
                    VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
                    return VK_SUCCESS;
                }
            }
        }

        res = blockVector.Allocate(
            size,
            alignment,
            finalCreateInfo,
            suballocType,
            allocationCount,
            pAllocations);
        if(res == VK_SUCCESS)
            return VK_SUCCESS;

        // Try dedicated memory.
        if(canAllocateDedicated && !dedicatedPreferred)
        {
            res = AllocateDedicatedMemory(
                pool,
                size,
                suballocType,
                dedicatedAllocations,
                memTypeIndex,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT) != 0,
                (finalCreateInfo.flags &
                    (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0,
                (finalCreateInfo.flags & VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT) != 0,
                finalCreateInfo.pUserData,
                finalCreateInfo.priority,
                dedicatedBuffer,
                dedicatedImage,
                dedicatedBufferImageUsage,
                allocationCount,
                pAllocations,
                blockVector.GetAllocationNextPtr());
            if(res == VK_SUCCESS)
            {
                // Succeeded: AllocateDedicatedMemory function already filled pMemory, nothing more to do here.
                VMA_DEBUG_LOG("    Allocated as DedicatedMemory");
                return VK_SUCCESS;
            }
        }
        // Everything failed: Return error code.
        VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
        return res;
    }
}

VkResult VmaAllocator_T::AllocateDedicatedMemory(
    VmaPool pool,
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    VmaDedicatedAllocationList& dedicatedAllocations,
    uint32_t memTypeIndex,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    bool canAliasMemory,
    void* pUserData,
    float priority,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VkFlags dedicatedBufferImageUsage,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    const void* pNextChain)
{
    VMA_ASSERT(allocationCount > 0 && pAllocations);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.memoryTypeIndex = memTypeIndex;
    allocInfo.allocationSize = size;
    allocInfo.pNext = pNextChain;

#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    VkMemoryDedicatedAllocateInfoKHR dedicatedAllocInfo = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR };
    if(!canAliasMemory)
    {
        if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
        {
            if(dedicatedBuffer != VK_NULL_HANDLE)
            {
                VMA_ASSERT(dedicatedImage == VK_NULL_HANDLE);
                dedicatedAllocInfo.buffer = dedicatedBuffer;
                VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
            else if(dedicatedImage != VK_NULL_HANDLE)
            {
                dedicatedAllocInfo.image = dedicatedImage;
                VmaPnextChainPushFront(&allocInfo, &dedicatedAllocInfo);
            }
        }
    }
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000

#if VMA_BUFFER_DEVICE_ADDRESS
    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR };
    if(m_UseKhrBufferDeviceAddress)
    {
        bool canContainBufferWithDeviceAddress = true;
        if(dedicatedBuffer != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = dedicatedBufferImageUsage == UINT32_MAX || // Usage flags unknown
                (dedicatedBufferImageUsage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT) != 0;
        }
        else if(dedicatedImage != VK_NULL_HANDLE)
        {
            canContainBufferWithDeviceAddress = false;
        }
        if(canContainBufferWithDeviceAddress)
        {
            allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
            VmaPnextChainPushFront(&allocInfo, &allocFlagsInfo);
        }
    }
#endif // #if VMA_BUFFER_DEVICE_ADDRESS

#if VMA_MEMORY_PRIORITY
    VkMemoryPriorityAllocateInfoEXT priorityInfo = { VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT };
    if(m_UseExtMemoryPriority)
    {
        VMA_ASSERT(priority >= 0.f && priority <= 1.f);
        priorityInfo.priority = priority;
        VmaPnextChainPushFront(&allocInfo, &priorityInfo);
    }
#endif // #if VMA_MEMORY_PRIORITY

#if VMA_EXTERNAL_MEMORY
    // Attach VkExportMemoryAllocateInfoKHR if necessary.
    VkExportMemoryAllocateInfoKHR exportMemoryAllocInfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR };
    exportMemoryAllocInfo.handleTypes = GetExternalMemoryHandleTypeFlags(memTypeIndex);
    if(exportMemoryAllocInfo.handleTypes != 0)
    {
        VmaPnextChainPushFront(&allocInfo, &exportMemoryAllocInfo);
    }
#endif // #if VMA_EXTERNAL_MEMORY

    size_t allocIndex;
    VkResult res = VK_SUCCESS;
    for(allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        res = AllocateDedicatedMemoryPage(
            pool,
            size,
            suballocType,
            memTypeIndex,
            allocInfo,
            map,
            isUserDataString,
            isMappingAllowed,
            pUserData,
            pAllocations + allocIndex);
        if(res != VK_SUCCESS)
        {
            break;
        }
    }

    if(res == VK_SUCCESS)
    {
        for (allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
        {
            dedicatedAllocations.Register(pAllocations[allocIndex]);
        }
        VMA_DEBUG_LOG_FORMAT("    Allocated DedicatedMemory Count=%zu, MemoryTypeIndex=#%u", allocationCount, memTypeIndex);
    }
    else
    {
        // Free all already created allocations.
        while(allocIndex--)
        {
            VmaAllocation currAlloc = pAllocations[allocIndex];
            VkDeviceMemory hMemory = currAlloc->GetMemory();

            /*
            There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
            before vkFreeMemory.

            if(currAlloc->GetMappedData() != VMA_NULL)
            {
                (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
            }
            */

            FreeVulkanMemory(memTypeIndex, currAlloc->GetSize(), hMemory);
            m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), currAlloc->GetSize());
            m_AllocationObjectAllocator.Free(currAlloc);
        }

        memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);
    }

    return res;
}

VkResult VmaAllocator_T::AllocateDedicatedMemoryPage(
    VmaPool pool,
    VkDeviceSize size,
    VmaSuballocationType suballocType,
    uint32_t memTypeIndex,
    const VkMemoryAllocateInfo& allocInfo,
    bool map,
    bool isUserDataString,
    bool isMappingAllowed,
    void* pUserData,
    VmaAllocation* pAllocation)
{
    VkDeviceMemory hMemory = VK_NULL_HANDLE;
    VkResult res = AllocateVulkanMemory(&allocInfo, &hMemory);
    if(res < 0)
    {
        VMA_DEBUG_LOG("    vkAllocateMemory FAILED");
        return res;
    }

    void* pMappedData = VMA_NULL;
    if(map)
    {
        res = (*m_VulkanFunctions.vkMapMemory)(
            m_hDevice,
            hMemory,
            0,
            VK_WHOLE_SIZE,
            0,
            &pMappedData);
        if(res < 0)
        {
            VMA_DEBUG_LOG("    vkMapMemory FAILED");
            FreeVulkanMemory(memTypeIndex, size, hMemory);
            return res;
        }
    }

    *pAllocation = m_AllocationObjectAllocator.Allocate(isMappingAllowed);
    (*pAllocation)->InitDedicatedAllocation(pool, memTypeIndex, hMemory, suballocType, pMappedData, size);
    if (isUserDataString)
        (*pAllocation)->SetName(this, (const char*)pUserData);
    else
        (*pAllocation)->SetUserData(this, pUserData);
    m_Budget.AddAllocation(MemoryTypeIndexToHeapIndex(memTypeIndex), size);
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
    {
        FillAllocation(*pAllocation, VMA_ALLOCATION_FILL_PATTERN_CREATED);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::GetBufferMemoryRequirements(
    VkBuffer hBuffer,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkBufferMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.buffer = hBuffer;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*m_VulkanFunctions.vkGetBufferMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetBufferMemoryRequirements)(m_hDevice, hBuffer, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

void VmaAllocator_T::GetImageMemoryRequirements(
    VkImage hImage,
    VkMemoryRequirements& memReq,
    bool& requiresDedicatedAllocation,
    bool& prefersDedicatedAllocation) const
{
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    if(m_UseKhrDedicatedAllocation || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0))
    {
        VkImageMemoryRequirementsInfo2KHR memReqInfo = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR };
        memReqInfo.image = hImage;

        VkMemoryDedicatedRequirementsKHR memDedicatedReq = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR };

        VkMemoryRequirements2KHR memReq2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR };
        VmaPnextChainPushFront(&memReq2, &memDedicatedReq);

        (*m_VulkanFunctions.vkGetImageMemoryRequirements2KHR)(m_hDevice, &memReqInfo, &memReq2);

        memReq = memReq2.memoryRequirements;
        requiresDedicatedAllocation = (memDedicatedReq.requiresDedicatedAllocation != VK_FALSE);
        prefersDedicatedAllocation  = (memDedicatedReq.prefersDedicatedAllocation  != VK_FALSE);
    }
    else
#endif // #if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
    {
        (*m_VulkanFunctions.vkGetImageMemoryRequirements)(m_hDevice, hImage, &memReq);
        requiresDedicatedAllocation = false;
        prefersDedicatedAllocation  = false;
    }
}

VkResult VmaAllocator_T::FindMemoryTypeIndex(
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkFlags bufImgUsage,
    uint32_t* pMemoryTypeIndex) const
{
    memoryTypeBits &= GetGlobalMemoryTypeBits();

    if(pAllocationCreateInfo->memoryTypeBits != 0)
    {
        memoryTypeBits &= pAllocationCreateInfo->memoryTypeBits;
    }

    VkMemoryPropertyFlags requiredFlags = 0, preferredFlags = 0, notPreferredFlags = 0;
    if(!FindMemoryPreferences(
        IsIntegratedGpu(),
        *pAllocationCreateInfo,
        bufImgUsage,
        requiredFlags, preferredFlags, notPreferredFlags))
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    *pMemoryTypeIndex = UINT32_MAX;
    uint32_t minCost = UINT32_MAX;
    for(uint32_t memTypeIndex = 0, memTypeBit = 1;
        memTypeIndex < GetMemoryTypeCount();
        ++memTypeIndex, memTypeBit <<= 1)
    {
        // This memory type is acceptable according to memoryTypeBits bitmask.
        if((memTypeBit & memoryTypeBits) != 0)
        {
            const VkMemoryPropertyFlags currFlags =
                m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
            // This memory type contains requiredFlags.
            if((requiredFlags & ~currFlags) == 0)
            {
                // Calculate cost as number of bits from preferredFlags not present in this memory type.
                uint32_t currCost = VMA_COUNT_BITS_SET(preferredFlags & ~currFlags) +
                    VMA_COUNT_BITS_SET(currFlags & notPreferredFlags);
                // Remember memory type with lowest cost.
                if(currCost < minCost)
                {
                    *pMemoryTypeIndex = memTypeIndex;
                    if(currCost == 0)
                    {
                        return VK_SUCCESS;
                    }
                    minCost = currCost;
                }
            }
        }
    }
    return (*pMemoryTypeIndex != UINT32_MAX) ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult VmaAllocator_T::CalcMemTypeParams(
    VmaAllocationCreateInfo& inoutCreateInfo,
    uint32_t memTypeIndex,
    VkDeviceSize size,
    size_t allocationCount)
{
    // If memory type is not HOST_VISIBLE, disable MAPPED.
    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0 &&
        (m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
    {
        inoutCreateInfo.flags &= ~VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) != 0)
    {
        const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memTypeIndex);
        VmaBudget heapBudget = {};
        GetHeapBudgets(&heapBudget, heapIndex, 1);
        if(heapBudget.usage + size * allocationCount > heapBudget.budget)
        {
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    return VK_SUCCESS;
}

VkResult VmaAllocator_T::CalcAllocationParams(
    VmaAllocationCreateInfo& inoutCreateInfo,
    bool dedicatedRequired,
    bool dedicatedPreferred)
{
    VMA_ASSERT((inoutCreateInfo.flags &
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) !=
        (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) &&
        "Specifying both flags VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT and VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT is incorrect.");
    VMA_ASSERT((((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT) == 0 ||
        (inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0)) &&
        "Specifying VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT requires also VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
    if(inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE || inoutCreateInfo.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) != 0)
        {
            VMA_ASSERT((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) != 0 &&
                "When using VMA_ALLOCATION_CREATE_MAPPED_BIT and usage = VMA_MEMORY_USAGE_AUTO*, you must also specify VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.");
        }
    }

    // If memory is lazily allocated, it should be always dedicated.
    if(dedicatedRequired ||
        inoutCreateInfo.usage == VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED)
    {
        inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    if(inoutCreateInfo.pool != VK_NULL_HANDLE)
    {
        if(inoutCreateInfo.pool->m_BlockVector.HasExplicitBlockSize() &&
            (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0)
        {
            VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT while current custom pool doesn't support dedicated allocations.");
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        inoutCreateInfo.priority = inoutCreateInfo.pool->m_BlockVector.GetPriority();
    }

    if((inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT) != 0 &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        VMA_ASSERT(0 && "Specifying VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT together with VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT makes no sense.");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    if(VMA_DEBUG_ALWAYS_DEDICATED_MEMORY &&
        (inoutCreateInfo.flags & VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT) != 0)
    {
        inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    // Non-auto USAGE values imply HOST_ACCESS flags.
    // And so does VMA_MEMORY_USAGE_UNKNOWN because it is used with custom pools.
    // Which specific flag is used doesn't matter. They change things only when used with VMA_MEMORY_USAGE_AUTO*.
    // Otherwise they just protect from assert on mapping.
    if(inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO &&
        inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE &&
        inoutCreateInfo.usage != VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
    {
        if((inoutCreateInfo.flags & (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) == 0)
        {
            inoutCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        }
    }

    return VK_SUCCESS;
}

VkResult VmaAllocator_T::AllocateMemory(
    const VkMemoryRequirements& vkMemReq,
    bool requiresDedicatedAllocation,
    bool prefersDedicatedAllocation,
    VkBuffer dedicatedBuffer,
    VkImage dedicatedImage,
    VkFlags dedicatedBufferImageUsage,
    const VmaAllocationCreateInfo& createInfo,
    VmaSuballocationType suballocType,
    size_t allocationCount,
    VmaAllocation* pAllocations)
{
    memset(pAllocations, 0, sizeof(VmaAllocation) * allocationCount);

    VMA_ASSERT(VmaIsPow2(vkMemReq.alignment));

    if(vkMemReq.size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VmaAllocationCreateInfo createInfoFinal = createInfo;
    VkResult res = CalcAllocationParams(createInfoFinal, requiresDedicatedAllocation, prefersDedicatedAllocation);
    if(res != VK_SUCCESS)
        return res;

    if(createInfoFinal.pool != VK_NULL_HANDLE)
    {
        VmaBlockVector& blockVector = createInfoFinal.pool->m_BlockVector;
        return AllocateMemoryOfType(
            createInfoFinal.pool,
            vkMemReq.size,
            vkMemReq.alignment,
            prefersDedicatedAllocation,
            dedicatedBuffer,
            dedicatedImage,
            dedicatedBufferImageUsage,
            createInfoFinal,
            blockVector.GetMemoryTypeIndex(),
            suballocType,
            createInfoFinal.pool->m_DedicatedAllocations,
            blockVector,
            allocationCount,
            pAllocations);
    }
    else
    {
        // Bit mask of memory Vulkan types acceptable for this allocation.
        uint32_t memoryTypeBits = vkMemReq.memoryTypeBits;
        uint32_t memTypeIndex = UINT32_MAX;
        res = FindMemoryTypeIndex(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
        // Can't find any single memory type matching requirements. res is VK_ERROR_FEATURE_NOT_PRESENT.
        if(res != VK_SUCCESS)
            return res;
        do
        {
            VmaBlockVector* blockVector = m_pBlockVectors[memTypeIndex];
            VMA_ASSERT(blockVector && "Trying to use unsupported memory type!");
            res = AllocateMemoryOfType(
                VK_NULL_HANDLE,
                vkMemReq.size,
                vkMemReq.alignment,
                requiresDedicatedAllocation || prefersDedicatedAllocation,
                dedicatedBuffer,
                dedicatedImage,
                dedicatedBufferImageUsage,
                createInfoFinal,
                memTypeIndex,
                suballocType,
                m_DedicatedAllocations[memTypeIndex],
                *blockVector,
                allocationCount,
                pAllocations);
            // Allocation succeeded
            if(res == VK_SUCCESS)
                return VK_SUCCESS;

            // Remove old memTypeIndex from list of possibilities.
            memoryTypeBits &= ~(1u << memTypeIndex);
            // Find alternative memTypeIndex.
            res = FindMemoryTypeIndex(memoryTypeBits, &createInfoFinal, dedicatedBufferImageUsage, &memTypeIndex);
        } while(res == VK_SUCCESS);

        // No other matching memory type index could be found.
        // Not returning res, which is VK_ERROR_FEATURE_NOT_PRESENT, because we already failed to allocate once.
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
}

void VmaAllocator_T::FreeMemory(
    size_t allocationCount,
    const VmaAllocation* pAllocations)
{
    VMA_ASSERT(pAllocations);

    for(size_t allocIndex = allocationCount; allocIndex--; )
    {
        VmaAllocation allocation = pAllocations[allocIndex];

        if(allocation != VK_NULL_HANDLE)
        {
            if(VMA_DEBUG_INITIALIZE_ALLOCATIONS)
            {
                FillAllocation(allocation, VMA_ALLOCATION_FILL_PATTERN_DESTROYED);
            }

            allocation->FreeName(this);

            switch(allocation->GetType())
            {
            case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
                {
                    VmaBlockVector* pBlockVector = VMA_NULL;
                    VmaPool hPool = allocation->GetParentPool();
                    if(hPool != VK_NULL_HANDLE)
                    {
                        pBlockVector = &hPool->m_BlockVector;
                    }
                    else
                    {
                        const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
                        pBlockVector = m_pBlockVectors[memTypeIndex];
                        VMA_ASSERT(pBlockVector && "Trying to free memory of unsupported type!");
                    }
                    pBlockVector->Free(allocation);
                }
                break;
            case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
                FreeDedicatedMemory(allocation);
                break;
            default:
                VMA_ASSERT(0);
            }
        }
    }
}

void VmaAllocator_T::CalculateStatistics(VmaTotalStatistics* pStats)
{
    // Initialize.
    VmaClearDetailedStatistics(pStats->total);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
        VmaClearDetailedStatistics(pStats->memoryType[i]);
    for(uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i)
        VmaClearDetailedStatistics(pStats->memoryHeap[i]);

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
        if (pBlockVector != VMA_NULL)
            pBlockVector->AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
        {
            VmaBlockVector& blockVector = pool->m_BlockVector;
            const uint32_t memTypeIndex = blockVector.GetMemoryTypeIndex();
            blockVector.AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
            pool->m_DedicatedAllocations.AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
        }
    }

    // Process dedicated allocations.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        m_DedicatedAllocations[memTypeIndex].AddDetailedStatistics(pStats->memoryType[memTypeIndex]);
    }

    // Sum from memory types to memory heaps.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        const uint32_t memHeapIndex = m_MemProps.memoryTypes[memTypeIndex].heapIndex;
        VmaAddDetailedStatistics(pStats->memoryHeap[memHeapIndex], pStats->memoryType[memTypeIndex]);
    }

    // Sum from memory heaps to total.
    for(uint32_t memHeapIndex = 0; memHeapIndex < GetMemoryHeapCount(); ++memHeapIndex)
        VmaAddDetailedStatistics(pStats->total, pStats->memoryHeap[memHeapIndex]);

    VMA_ASSERT(pStats->total.statistics.allocationCount == 0 ||
        pStats->total.allocationSizeMax >= pStats->total.allocationSizeMin);
    VMA_ASSERT(pStats->total.unusedRangeCount == 0 ||
        pStats->total.unusedRangeSizeMax >= pStats->total.unusedRangeSizeMin);
}

void VmaAllocator_T::GetHeapBudgets(VmaBudget* outBudgets, uint32_t firstHeap, uint32_t heapCount)
{
#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        if(m_Budget.m_OperationsSinceBudgetFetch < 30)
        {
            VmaMutexLockRead lockRead(m_Budget.m_BudgetMutex, m_UseMutex);
            for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
            {
                const uint32_t heapIndex = firstHeap + i;

                outBudgets->statistics.blockCount = m_Budget.m_BlockCount[heapIndex];
                outBudgets->statistics.allocationCount = m_Budget.m_AllocationCount[heapIndex];
                outBudgets->statistics.blockBytes = m_Budget.m_BlockBytes[heapIndex];
                outBudgets->statistics.allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

                if(m_Budget.m_VulkanUsage[heapIndex] + outBudgets->statistics.blockBytes > m_Budget.m_BlockBytesAtBudgetFetch[heapIndex])
                {
                    outBudgets->usage = m_Budget.m_VulkanUsage[heapIndex] +
                        outBudgets->statistics.blockBytes - m_Budget.m_BlockBytesAtBudgetFetch[heapIndex];
                }
                else
                {
                    outBudgets->usage = 0;
                }

                // Have to take MIN with heap size because explicit HeapSizeLimit is included in it.
                outBudgets->budget = VMA_MIN(
                    m_Budget.m_VulkanBudget[heapIndex], m_MemProps.memoryHeaps[heapIndex].size);
            }
        }
        else
        {
            UpdateVulkanBudget(); // Outside of mutex lock
            GetHeapBudgets(outBudgets, firstHeap, heapCount); // Recursion
        }
    }
    else
#endif
    {
        for(uint32_t i = 0; i < heapCount; ++i, ++outBudgets)
        {
            const uint32_t heapIndex = firstHeap + i;

            outBudgets->statistics.blockCount = m_Budget.m_BlockCount[heapIndex];
            outBudgets->statistics.allocationCount = m_Budget.m_AllocationCount[heapIndex];
            outBudgets->statistics.blockBytes = m_Budget.m_BlockBytes[heapIndex];
            outBudgets->statistics.allocationBytes = m_Budget.m_AllocationBytes[heapIndex];

            outBudgets->usage = outBudgets->statistics.blockBytes;
            outBudgets->budget = m_MemProps.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
        }
    }
}

void VmaAllocator_T::GetAllocationInfo(VmaAllocation hAllocation, VmaAllocationInfo* pAllocationInfo)
{
    pAllocationInfo->memoryType = hAllocation->GetMemoryTypeIndex();
    pAllocationInfo->deviceMemory = hAllocation->GetMemory();
    pAllocationInfo->offset = hAllocation->GetOffset();
    pAllocationInfo->size = hAllocation->GetSize();
    pAllocationInfo->pMappedData = hAllocation->GetMappedData();
    pAllocationInfo->pUserData = hAllocation->GetUserData();
    pAllocationInfo->pName = hAllocation->GetName();
}

VkResult VmaAllocator_T::CreatePool(const VmaPoolCreateInfo* pCreateInfo, VmaPool* pPool)
{
    VMA_DEBUG_LOG_FORMAT("  CreatePool: MemoryTypeIndex=%u, flags=%u", pCreateInfo->memoryTypeIndex, pCreateInfo->flags);

    VmaPoolCreateInfo newCreateInfo = *pCreateInfo;

    // Protection against uninitialized new structure member. If garbage data are left there, this pointer dereference would crash.
    if(pCreateInfo->pMemoryAllocateNext)
    {
        VMA_ASSERT(((const VkBaseInStructure*)pCreateInfo->pMemoryAllocateNext)->sType != 0);
    }

    if(newCreateInfo.maxBlockCount == 0)
    {
        newCreateInfo.maxBlockCount = SIZE_MAX;
    }
    if(newCreateInfo.minBlockCount > newCreateInfo.maxBlockCount)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    // Memory type index out of range or forbidden.
    if(pCreateInfo->memoryTypeIndex >= GetMemoryTypeCount() ||
        ((1u << pCreateInfo->memoryTypeIndex) & m_GlobalMemoryTypeBits) == 0)
    {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if(newCreateInfo.minAllocationAlignment > 0)
    {
        VMA_ASSERT(VmaIsPow2(newCreateInfo.minAllocationAlignment));
    }

    const VkDeviceSize preferredBlockSize = CalcPreferredBlockSize(newCreateInfo.memoryTypeIndex);

    *pPool = vma_new(this, VmaPool_T)(this, newCreateInfo, preferredBlockSize);

    VkResult res = (*pPool)->m_BlockVector.CreateMinBlocks();
    if(res != VK_SUCCESS)
    {
        vma_delete(this, *pPool);
        *pPool = VMA_NULL;
        return res;
    }

    // Add to m_Pools.
    {
        VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        (*pPool)->SetId(m_NextPoolId++);
        m_Pools.PushBack(*pPool);
    }

    return VK_SUCCESS;
}

void VmaAllocator_T::DestroyPool(VmaPool pool)
{
    // Remove from m_Pools.
    {
        VmaMutexLockWrite lock(m_PoolsMutex, m_UseMutex);
        m_Pools.Remove(pool);
    }

    vma_delete(this, pool);
}

void VmaAllocator_T::GetPoolStatistics(VmaPool pool, VmaStatistics* pPoolStats)
{
    VmaClearStatistics(*pPoolStats);
    pool->m_BlockVector.AddStatistics(*pPoolStats);
    pool->m_DedicatedAllocations.AddStatistics(*pPoolStats);
}

void VmaAllocator_T::CalculatePoolStatistics(VmaPool pool, VmaDetailedStatistics* pPoolStats)
{
    VmaClearDetailedStatistics(*pPoolStats);
    pool->m_BlockVector.AddDetailedStatistics(*pPoolStats);
    pool->m_DedicatedAllocations.AddDetailedStatistics(*pPoolStats);
}

void VmaAllocator_T::SetCurrentFrameIndex(uint32_t frameIndex)
{
    m_CurrentFrameIndex.store(frameIndex);

#if VMA_MEMORY_BUDGET
    if(m_UseExtMemoryBudget)
    {
        UpdateVulkanBudget();
    }
#endif // #if VMA_MEMORY_BUDGET
}

VkResult VmaAllocator_T::CheckPoolCorruption(VmaPool hPool)
{
    return hPool->m_BlockVector.CheckCorruption();
}

VkResult VmaAllocator_T::CheckCorruption(uint32_t memoryTypeBits)
{
    VkResult finalRes = VK_ERROR_FEATURE_NOT_PRESENT;

    // Process default pools.
    for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
    {
        VmaBlockVector* const pBlockVector = m_pBlockVectors[memTypeIndex];
        if(pBlockVector != VMA_NULL)
        {
            VkResult localRes = pBlockVector->CheckCorruption();
            switch(localRes)
            {
            case VK_ERROR_FEATURE_NOT_PRESENT:
                break;
            case VK_SUCCESS:
                finalRes = VK_SUCCESS;
                break;
            default:
                return localRes;
            }
        }
    }

    // Process custom pools.
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        for(VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
        {
            if(((1u << pool->m_BlockVector.GetMemoryTypeIndex()) & memoryTypeBits) != 0)
            {
                VkResult localRes = pool->m_BlockVector.CheckCorruption();
                switch(localRes)
                {
                case VK_ERROR_FEATURE_NOT_PRESENT:
                    break;
                case VK_SUCCESS:
                    finalRes = VK_SUCCESS;
                    break;
                default:
                    return localRes;
                }
            }
        }
    }

    return finalRes;
}

VkResult VmaAllocator_T::AllocateVulkanMemory(const VkMemoryAllocateInfo* pAllocateInfo, VkDeviceMemory* pMemory)
{
    AtomicTransactionalIncrement<VMA_ATOMIC_UINT32> deviceMemoryCountIncrement;
    const uint64_t prevDeviceMemoryCount = deviceMemoryCountIncrement.Increment(&m_DeviceMemoryCount);
#if VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
    if(prevDeviceMemoryCount >= m_PhysicalDeviceProperties.limits.maxMemoryAllocationCount)
    {
        return VK_ERROR_TOO_MANY_OBJECTS;
    }
#endif

    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(pAllocateInfo->memoryTypeIndex);

    // HeapSizeLimit is in effect for this heap.
    if((m_HeapSizeLimitMask & (1u << heapIndex)) != 0)
    {
        const VkDeviceSize heapSize = m_MemProps.memoryHeaps[heapIndex].size;
        VkDeviceSize blockBytes = m_Budget.m_BlockBytes[heapIndex];
        for(;;)
        {
            const VkDeviceSize blockBytesAfterAllocation = blockBytes + pAllocateInfo->allocationSize;
            if(blockBytesAfterAllocation > heapSize)
            {
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            }
            if(m_Budget.m_BlockBytes[heapIndex].compare_exchange_strong(blockBytes, blockBytesAfterAllocation))
            {
                break;
            }
        }
    }
    else
    {
        m_Budget.m_BlockBytes[heapIndex] += pAllocateInfo->allocationSize;
    }
    ++m_Budget.m_BlockCount[heapIndex];

    // VULKAN CALL vkAllocateMemory.
    VkResult res = (*m_VulkanFunctions.vkAllocateMemory)(m_hDevice, pAllocateInfo, GetAllocationCallbacks(), pMemory);

    if(res == VK_SUCCESS)
    {
#if VMA_MEMORY_BUDGET
        ++m_Budget.m_OperationsSinceBudgetFetch;
#endif

        // Informative callback.
        if(m_DeviceMemoryCallbacks.pfnAllocate != VMA_NULL)
        {
            (*m_DeviceMemoryCallbacks.pfnAllocate)(this, pAllocateInfo->memoryTypeIndex, *pMemory, pAllocateInfo->allocationSize, m_DeviceMemoryCallbacks.pUserData);
        }

        deviceMemoryCountIncrement.Commit();
    }
    else
    {
        --m_Budget.m_BlockCount[heapIndex];
        m_Budget.m_BlockBytes[heapIndex] -= pAllocateInfo->allocationSize;
    }

    return res;
}

void VmaAllocator_T::FreeVulkanMemory(uint32_t memoryType, VkDeviceSize size, VkDeviceMemory hMemory)
{
    // Informative callback.
    if(m_DeviceMemoryCallbacks.pfnFree != VMA_NULL)
    {
        (*m_DeviceMemoryCallbacks.pfnFree)(this, memoryType, hMemory, size, m_DeviceMemoryCallbacks.pUserData);
    }

    // VULKAN CALL vkFreeMemory.
    (*m_VulkanFunctions.vkFreeMemory)(m_hDevice, hMemory, GetAllocationCallbacks());

    const uint32_t heapIndex = MemoryTypeIndexToHeapIndex(memoryType);
    --m_Budget.m_BlockCount[heapIndex];
    m_Budget.m_BlockBytes[heapIndex] -= size;

    --m_DeviceMemoryCount;
}

VkResult VmaAllocator_T::BindVulkanBuffer(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkBuffer buffer,
    const void* pNext)
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindBufferMemory2KHR != VMA_NULL)
        {
            VkBindBufferMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.buffer = buffer;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindBufferMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
        else
#endif // #if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    else
    {
        return (*m_VulkanFunctions.vkBindBufferMemory)(m_hDevice, buffer, memory, memoryOffset);
    }
}

VkResult VmaAllocator_T::BindVulkanImage(
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset,
    VkImage image,
    const void* pNext)
{
    if(pNext != VMA_NULL)
    {
#if VMA_VULKAN_VERSION >= 1001000 || VMA_BIND_MEMORY2
        if((m_UseKhrBindMemory2 || m_VulkanApiVersion >= VK_MAKE_VERSION(1, 1, 0)) &&
            m_VulkanFunctions.vkBindImageMemory2KHR != VMA_NULL)
        {
            VkBindImageMemoryInfoKHR bindBufferMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR };
            bindBufferMemoryInfo.pNext = pNext;
            bindBufferMemoryInfo.image = image;
            bindBufferMemoryInfo.memory = memory;
            bindBufferMemoryInfo.memoryOffset = memoryOffset;
            return (*m_VulkanFunctions.vkBindImageMemory2KHR)(m_hDevice, 1, &bindBufferMemoryInfo);
        }
        else
#endif // #if VMA_BIND_MEMORY2
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    else
    {
        return (*m_VulkanFunctions.vkBindImageMemory)(m_hDevice, image, memory, memoryOffset);
    }
}

VkResult VmaAllocator_T::Map(VmaAllocation hAllocation, void** ppData)
{
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            char *pBytes = VMA_NULL;
            VkResult res = pBlock->Map(this, 1, (void**)&pBytes);
            if(res == VK_SUCCESS)
            {
                *ppData = pBytes + (ptrdiff_t)hAllocation->GetOffset();
                hAllocation->BlockAllocMap();
            }
            return res;
        }
        VMA_FALLTHROUGH; // Fallthrough
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        return hAllocation->DedicatedAllocMap(this, ppData);
    default:
        VMA_ASSERT(0);
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
}

void VmaAllocator_T::Unmap(VmaAllocation hAllocation)
{
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
            hAllocation->BlockAllocUnmap();
            pBlock->Unmap(this, 1);
        }
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        hAllocation->DedicatedAllocUnmap(this);
        break;
    default:
        VMA_ASSERT(0);
    }
}

VkResult VmaAllocator_T::BindBufferMemory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer hBuffer,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN;
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanBuffer(hAllocation->GetMemory(), allocationLocalOffset, hBuffer, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* const pBlock = hAllocation->GetBlock();
        VMA_ASSERT(pBlock && "Binding buffer to allocation that doesn't belong to any block.");
        res = pBlock->BindBufferMemory(this, hAllocation, allocationLocalOffset, hBuffer, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

VkResult VmaAllocator_T::BindImageMemory(
    VmaAllocation hAllocation,
    VkDeviceSize allocationLocalOffset,
    VkImage hImage,
    const void* pNext)
{
    VkResult res = VK_ERROR_UNKNOWN;
    switch(hAllocation->GetType())
    {
    case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
        res = BindVulkanImage(hAllocation->GetMemory(), allocationLocalOffset, hImage, pNext);
        break;
    case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
    {
        VmaDeviceMemoryBlock* pBlock = hAllocation->GetBlock();
        VMA_ASSERT(pBlock && "Binding image to allocation that doesn't belong to any block.");
        res = pBlock->BindImageMemory(this, hAllocation, allocationLocalOffset, hImage, pNext);
        break;
    }
    default:
        VMA_ASSERT(0);
    }
    return res;
}

VkResult VmaAllocator_T::FlushOrInvalidateAllocation(
    VmaAllocation hAllocation,
    VkDeviceSize offset, VkDeviceSize size,
    VMA_CACHE_OPERATION op)
{
    VkResult res = VK_SUCCESS;

    VkMappedMemoryRange memRange = {};
    if(GetFlushOrInvalidateRange(hAllocation, offset, size, memRange))
    {
        switch(op)
        {
        case VMA_CACHE_FLUSH:
            res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        case VMA_CACHE_INVALIDATE:
            res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, 1, &memRange);
            break;
        default:
            VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

VkResult VmaAllocator_T::FlushOrInvalidateAllocations(
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets, const VkDeviceSize* sizes,
    VMA_CACHE_OPERATION op)
{
    typedef VmaStlAllocator<VkMappedMemoryRange> RangeAllocator;
    typedef VmaSmallVector<VkMappedMemoryRange, RangeAllocator, 16> RangeVector;
    RangeVector ranges = RangeVector(RangeAllocator(GetAllocationCallbacks()));

    for(uint32_t allocIndex = 0; allocIndex < allocationCount; ++allocIndex)
    {
        const VmaAllocation alloc = allocations[allocIndex];
        const VkDeviceSize offset = offsets != VMA_NULL ? offsets[allocIndex] : 0;
        const VkDeviceSize size = sizes != VMA_NULL ? sizes[allocIndex] : VK_WHOLE_SIZE;
        VkMappedMemoryRange newRange;
        if(GetFlushOrInvalidateRange(alloc, offset, size, newRange))
        {
            ranges.push_back(newRange);
        }
    }

    VkResult res = VK_SUCCESS;
    if(!ranges.empty())
    {
        switch(op)
        {
        case VMA_CACHE_FLUSH:
            res = (*GetVulkanFunctions().vkFlushMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
            break;
        case VMA_CACHE_INVALIDATE:
            res = (*GetVulkanFunctions().vkInvalidateMappedMemoryRanges)(m_hDevice, (uint32_t)ranges.size(), ranges.data());
            break;
        default:
            VMA_ASSERT(0);
        }
    }
    // else: Just ignore this call.
    return res;
}

void VmaAllocator_T::FreeDedicatedMemory(const VmaAllocation allocation)
{
    VMA_ASSERT(allocation && allocation->GetType() == VmaAllocation_T::ALLOCATION_TYPE_DEDICATED);

    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    VmaPool parentPool = allocation->GetParentPool();
    if(parentPool == VK_NULL_HANDLE)
    {
        // Default pool
        m_DedicatedAllocations[memTypeIndex].Unregister(allocation);
    }
    else
    {
        // Custom pool
        parentPool->m_DedicatedAllocations.Unregister(allocation);
    }

    VkDeviceMemory hMemory = allocation->GetMemory();

    /*
    There is no need to call this, because Vulkan spec allows to skip vkUnmapMemory
    before vkFreeMemory.

    if(allocation->GetMappedData() != VMA_NULL)
    {
        (*m_VulkanFunctions.vkUnmapMemory)(m_hDevice, hMemory);
    }
    */

    FreeVulkanMemory(memTypeIndex, allocation->GetSize(), hMemory);

    m_Budget.RemoveAllocation(MemoryTypeIndexToHeapIndex(allocation->GetMemoryTypeIndex()), allocation->GetSize());
    m_AllocationObjectAllocator.Free(allocation);

    VMA_DEBUG_LOG_FORMAT("    Freed DedicatedMemory MemoryTypeIndex=%u", memTypeIndex);
}

uint32_t VmaAllocator_T::CalculateGpuDefragmentationMemoryTypeBits() const
{
    VkBufferCreateInfo dummyBufCreateInfo;
    VmaFillGpuDefragmentationBufferCreateInfo(dummyBufCreateInfo);

    uint32_t memoryTypeBits = 0;

    // Create buffer.
    VkBuffer buf = VK_NULL_HANDLE;
    VkResult res = (*GetVulkanFunctions().vkCreateBuffer)(
        m_hDevice, &dummyBufCreateInfo, GetAllocationCallbacks(), &buf);
    if(res == VK_SUCCESS)
    {
        // Query for supported memory types.
        VkMemoryRequirements memReq;
        (*GetVulkanFunctions().vkGetBufferMemoryRequirements)(m_hDevice, buf, &memReq);
        memoryTypeBits = memReq.memoryTypeBits;

        // Destroy buffer.
        (*GetVulkanFunctions().vkDestroyBuffer)(m_hDevice, buf, GetAllocationCallbacks());
    }

    return memoryTypeBits;
}

uint32_t VmaAllocator_T::CalculateGlobalMemoryTypeBits() const
{
    // Make sure memory information is already fetched.
    VMA_ASSERT(GetMemoryTypeCount() > 0);

    uint32_t memoryTypeBits = UINT32_MAX;

    if(!m_UseAmdDeviceCoherentMemory)
    {
        // Exclude memory types that have VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD.
        for(uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
        {
            if((m_MemProps.memoryTypes[memTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY) != 0)
            {
                memoryTypeBits &= ~(1u << memTypeIndex);
            }
        }
    }

    return memoryTypeBits;
}

bool VmaAllocator_T::GetFlushOrInvalidateRange(
    VmaAllocation allocation,
    VkDeviceSize offset, VkDeviceSize size,
    VkMappedMemoryRange& outRange) const
{
    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    if(size > 0 && IsMemoryTypeNonCoherent(memTypeIndex))
    {
        const VkDeviceSize nonCoherentAtomSize = m_PhysicalDeviceProperties.limits.nonCoherentAtomSize;
        const VkDeviceSize allocationSize = allocation->GetSize();
        VMA_ASSERT(offset <= allocationSize);

        outRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        outRange.pNext = VMA_NULL;
        outRange.memory = allocation->GetMemory();

        switch(allocation->GetType())
        {
        case VmaAllocation_T::ALLOCATION_TYPE_DEDICATED:
            outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                outRange.size = allocationSize - outRange.offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
                outRange.size = VMA_MIN(
                    VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize),
                    allocationSize - outRange.offset);
            }
            break;
        case VmaAllocation_T::ALLOCATION_TYPE_BLOCK:
        {
            // 1. Still within this allocation.
            outRange.offset = VmaAlignDown(offset, nonCoherentAtomSize);
            if(size == VK_WHOLE_SIZE)
            {
                size = allocationSize - offset;
            }
            else
            {
                VMA_ASSERT(offset + size <= allocationSize);
            }
            outRange.size = VmaAlignUp(size + (offset - outRange.offset), nonCoherentAtomSize);

            // 2. Adjust to whole block.
            const VkDeviceSize allocationOffset = allocation->GetOffset();
            VMA_ASSERT(allocationOffset % nonCoherentAtomSize == 0);
            const VkDeviceSize blockSize = allocation->GetBlock()->m_pMetadata->GetSize();
            outRange.offset += allocationOffset;
            outRange.size = VMA_MIN(outRange.size, blockSize - outRange.offset);

            break;
        }
        default:
            VMA_ASSERT(0);
        }
        return true;
    }
    return false;
}

#if VMA_MEMORY_BUDGET
void VmaAllocator_T::UpdateVulkanBudget()
{
    VMA_ASSERT(m_UseExtMemoryBudget);

    VkPhysicalDeviceMemoryProperties2KHR memProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR };

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budgetProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT };
    VmaPnextChainPushFront(&memProps, &budgetProps);

    GetVulkanFunctions().vkGetPhysicalDeviceMemoryProperties2KHR(m_PhysicalDevice, &memProps);

    {
        VmaMutexLockWrite lockWrite(m_Budget.m_BudgetMutex, m_UseMutex);

        for(uint32_t heapIndex = 0; heapIndex < GetMemoryHeapCount(); ++heapIndex)
        {
            m_Budget.m_VulkanUsage[heapIndex] = budgetProps.heapUsage[heapIndex];
            m_Budget.m_VulkanBudget[heapIndex] = budgetProps.heapBudget[heapIndex];
            m_Budget.m_BlockBytesAtBudgetFetch[heapIndex] = m_Budget.m_BlockBytes[heapIndex].load();

            // Some bugged drivers return the budget incorrectly, e.g. 0 or much bigger than heap size.
            if(m_Budget.m_VulkanBudget[heapIndex] == 0)
            {
                m_Budget.m_VulkanBudget[heapIndex] = m_MemProps.memoryHeaps[heapIndex].size * 8 / 10; // 80% heuristics.
            }
            else if(m_Budget.m_VulkanBudget[heapIndex] > m_MemProps.memoryHeaps[heapIndex].size)
            {
                m_Budget.m_VulkanBudget[heapIndex] = m_MemProps.memoryHeaps[heapIndex].size;
            }
            if(m_Budget.m_VulkanUsage[heapIndex] == 0 && m_Budget.m_BlockBytesAtBudgetFetch[heapIndex] > 0)
            {
                m_Budget.m_VulkanUsage[heapIndex] = m_Budget.m_BlockBytesAtBudgetFetch[heapIndex];
            }
        }
        m_Budget.m_OperationsSinceBudgetFetch = 0;
    }
}
#endif // VMA_MEMORY_BUDGET

void VmaAllocator_T::FillAllocation(const VmaAllocation hAllocation, uint8_t pattern)
{
    if(VMA_DEBUG_INITIALIZE_ALLOCATIONS &&
        hAllocation->IsMappingAllowed() &&
        (m_MemProps.memoryTypes[hAllocation->GetMemoryTypeIndex()].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
    {
        void* pData = VMA_NULL;
        VkResult res = Map(hAllocation, &pData);
        if(res == VK_SUCCESS)
        {
            memset(pData, (int)pattern, (size_t)hAllocation->GetSize());
            FlushOrInvalidateAllocation(hAllocation, 0, VK_WHOLE_SIZE, VMA_CACHE_FLUSH);
            Unmap(hAllocation);
        }
        else
        {
            VMA_ASSERT(0 && "VMA_DEBUG_INITIALIZE_ALLOCATIONS is enabled, but couldn't map memory to fill allocation.");
        }
    }
}

uint32_t VmaAllocator_T::GetGpuDefragmentationMemoryTypeBits()
{
    uint32_t memoryTypeBits = m_GpuDefragmentationMemoryTypeBits.load();
    if(memoryTypeBits == UINT32_MAX)
    {
        memoryTypeBits = CalculateGpuDefragmentationMemoryTypeBits();
        m_GpuDefragmentationMemoryTypeBits.store(memoryTypeBits);
    }
    return memoryTypeBits;
}

#if VMA_STATS_STRING_ENABLED
void VmaAllocator_T::PrintDetailedMap(VmaJsonWriter& json)
{
    json.WriteString("DefaultPools");
    json.BeginObject();
    {
        for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
        {
            VmaBlockVector* pBlockVector = m_pBlockVectors[memTypeIndex];
            VmaDedicatedAllocationList& dedicatedAllocList = m_DedicatedAllocations[memTypeIndex];
            if (pBlockVector != VMA_NULL)
            {
                json.BeginString("Type ");
                json.ContinueString(memTypeIndex);
                json.EndString();
                json.BeginObject();
                {
                    json.WriteString("PreferredBlockSize");
                    json.WriteNumber(pBlockVector->GetPreferredBlockSize());

                    json.WriteString("Blocks");
                    pBlockVector->PrintDetailedMap(json);

                    json.WriteString("DedicatedAllocations");
                    dedicatedAllocList.BuildStatsString(json);
                }
                json.EndObject();
            }
        }
    }
    json.EndObject();

    json.WriteString("CustomPools");
    json.BeginObject();
    {
        VmaMutexLockRead lock(m_PoolsMutex, m_UseMutex);
        if (!m_Pools.IsEmpty())
        {
            for (uint32_t memTypeIndex = 0; memTypeIndex < GetMemoryTypeCount(); ++memTypeIndex)
            {
                bool displayType = true;
                size_t index = 0;
                for (VmaPool pool = m_Pools.Front(); pool != VMA_NULL; pool = m_Pools.GetNext(pool))
                {
                    VmaBlockVector& blockVector = pool->m_BlockVector;
                    if (blockVector.GetMemoryTypeIndex() == memTypeIndex)
                    {
                        if (displayType)
                        {
                            json.BeginString("Type ");
                            json.ContinueString(memTypeIndex);
                            json.EndString();
                            json.BeginArray();
                            displayType = false;
                        }

                        json.BeginObject();
                        {
                            json.WriteString("Name");
                            json.BeginString();
                            json.ContinueString((uint64_t)index++);
                            if (pool->GetName())
                            {
                                json.ContinueString(" - ");
                                json.ContinueString(pool->GetName());
                            }
                            json.EndString();

                            json.WriteString("PreferredBlockSize");
                            json.WriteNumber(blockVector.GetPreferredBlockSize());

                            json.WriteString("Blocks");
                            blockVector.PrintDetailedMap(json);

                            json.WriteString("DedicatedAllocations");
                            pool->m_DedicatedAllocations.BuildStatsString(json);
                        }
                        json.EndObject();
                    }
                }

                if (!displayType)
                    json.EndArray();
            }
        }
    }
    json.EndObject();
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_ALLOCATOR_T_FUNCTIONS


#ifndef _VMA_PUBLIC_INTERFACE
VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
    const VmaAllocatorCreateInfo* pCreateInfo,
    VmaAllocator* pAllocator)
{
    VMA_ASSERT(pCreateInfo && pAllocator);
    VMA_ASSERT(pCreateInfo->vulkanApiVersion == 0 ||
        (VK_VERSION_MAJOR(pCreateInfo->vulkanApiVersion) == 1 && VK_VERSION_MINOR(pCreateInfo->vulkanApiVersion) <= 3));
    VMA_DEBUG_LOG("vmaCreateAllocator");
    *pAllocator = vma_new(pCreateInfo->pAllocationCallbacks, VmaAllocator_T)(pCreateInfo);
    VkResult result = (*pAllocator)->Init(pCreateInfo);
    if(result < 0)
    {
        vma_delete(pCreateInfo->pAllocationCallbacks, *pAllocator);
        *pAllocator = VK_NULL_HANDLE;
    }
    return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
    VmaAllocator allocator)
{
    if(allocator != VK_NULL_HANDLE)
    {
        VMA_DEBUG_LOG("vmaDestroyAllocator");
        VkAllocationCallbacks allocationCallbacks = allocator->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
        vma_delete(&allocationCallbacks, allocator);
    }
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocatorInfo(VmaAllocator allocator, VmaAllocatorInfo* pAllocatorInfo)
{
    VMA_ASSERT(allocator && pAllocatorInfo);
    pAllocatorInfo->instance = allocator->m_hInstance;
    pAllocatorInfo->physicalDevice = allocator->GetPhysicalDevice();
    pAllocatorInfo->device = allocator->m_hDevice;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceProperties **ppPhysicalDeviceProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceProperties);
    *ppPhysicalDeviceProperties = &allocator->m_PhysicalDeviceProperties;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
    VmaAllocator allocator,
    const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties)
{
    VMA_ASSERT(allocator && ppPhysicalDeviceMemoryProperties);
    *ppPhysicalDeviceMemoryProperties = &allocator->m_MemProps;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
    VmaAllocator allocator,
    uint32_t memoryTypeIndex,
    VkMemoryPropertyFlags* pFlags)
{
    VMA_ASSERT(allocator && pFlags);
    VMA_ASSERT(memoryTypeIndex < allocator->GetMemoryTypeCount());
    *pFlags = allocator->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags;
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
    VmaAllocator allocator,
    uint32_t frameIndex)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->SetCurrentFrameIndex(frameIndex);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStatistics(
    VmaAllocator allocator,
    VmaTotalStatistics* pStats)
{
    VMA_ASSERT(allocator && pStats);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->CalculateStatistics(pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetHeapBudgets(
    VmaAllocator allocator,
    VmaBudget* pBudgets)
{
    VMA_ASSERT(allocator && pBudgets);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK
    allocator->GetHeapBudgets(pBudgets, 0, allocator->GetMemoryHeapCount());
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
    VmaAllocator allocator,
    char** ppStatsString,
    VkBool32 detailedMap)
{
    VMA_ASSERT(allocator && ppStatsString);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VmaStringBuilder sb(allocator->GetAllocationCallbacks());
    {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        allocator->GetHeapBudgets(budgets, 0, allocator->GetMemoryHeapCount());

        VmaTotalStatistics stats;
        allocator->CalculateStatistics(&stats);

        VmaJsonWriter json(allocator->GetAllocationCallbacks(), sb);
        json.BeginObject();
        {
            json.WriteString("General");
            json.BeginObject();
            {
                const VkPhysicalDeviceProperties& deviceProperties = allocator->m_PhysicalDeviceProperties;
                const VkPhysicalDeviceMemoryProperties& memoryProperties = allocator->m_MemProps;

                json.WriteString("API");
                json.WriteString("Vulkan");

                json.WriteString("apiVersion");
                json.BeginString();
                json.ContinueString(VK_VERSION_MAJOR(deviceProperties.apiVersion));
                json.ContinueString(".");
                json.ContinueString(VK_VERSION_MINOR(deviceProperties.apiVersion));
                json.ContinueString(".");
                json.ContinueString(VK_VERSION_PATCH(deviceProperties.apiVersion));
                json.EndString();

                json.WriteString("GPU");
                json.WriteString(deviceProperties.deviceName);
                json.WriteString("deviceType");
                json.WriteNumber(static_cast<uint32_t>(deviceProperties.deviceType));

                json.WriteString("maxMemoryAllocationCount");
                json.WriteNumber(deviceProperties.limits.maxMemoryAllocationCount);
                json.WriteString("bufferImageGranularity");
                json.WriteNumber(deviceProperties.limits.bufferImageGranularity);
                json.WriteString("nonCoherentAtomSize");
                json.WriteNumber(deviceProperties.limits.nonCoherentAtomSize);

                json.WriteString("memoryHeapCount");
                json.WriteNumber(memoryProperties.memoryHeapCount);
                json.WriteString("memoryTypeCount");
                json.WriteNumber(memoryProperties.memoryTypeCount);
            }
            json.EndObject();
        }
        {
            json.WriteString("Total");
            VmaPrintDetailedStatistics(json, stats.total);
        }
        {
            json.WriteString("MemoryInfo");
            json.BeginObject();
            {
                for (uint32_t heapIndex = 0; heapIndex < allocator->GetMemoryHeapCount(); ++heapIndex)
                {
                    json.BeginString("Heap ");
                    json.ContinueString(heapIndex);
                    json.EndString();
                    json.BeginObject();
                    {
                        const VkMemoryHeap& heapInfo = allocator->m_MemProps.memoryHeaps[heapIndex];
                        json.WriteString("Flags");
                        json.BeginArray(true);
                        {
                            if (heapInfo.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                                json.WriteString("DEVICE_LOCAL");
                        #if VMA_VULKAN_VERSION >= 1001000
                            if (heapInfo.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
                                json.WriteString("MULTI_INSTANCE");
                        #endif

                            VkMemoryHeapFlags flags = heapInfo.flags &
                                ~(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                        #if VMA_VULKAN_VERSION >= 1001000
                                    | VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
                        #endif
                                    );
                            if (flags != 0)
                                json.WriteNumber(flags);
                        }
                        json.EndArray();

                        json.WriteString("Size");
                        json.WriteNumber(heapInfo.size);

                        json.WriteString("Budget");
                        json.BeginObject();
                        {
                            json.WriteString("BudgetBytes");
                            json.WriteNumber(budgets[heapIndex].budget);
                            json.WriteString("UsageBytes");
                            json.WriteNumber(budgets[heapIndex].usage);
                        }
                        json.EndObject();

                        json.WriteString("Stats");
                        VmaPrintDetailedStatistics(json, stats.memoryHeap[heapIndex]);

                        json.WriteString("MemoryPools");
                        json.BeginObject();
                        {
                            for (uint32_t typeIndex = 0; typeIndex < allocator->GetMemoryTypeCount(); ++typeIndex)
                            {
                                if (allocator->MemoryTypeIndexToHeapIndex(typeIndex) == heapIndex)
                                {
                                    json.BeginString("Type ");
                                    json.ContinueString(typeIndex);
                                    json.EndString();
                                    json.BeginObject();
                                    {
                                        json.WriteString("Flags");
                                        json.BeginArray(true);
                                        {
                                            VkMemoryPropertyFlags flags = allocator->m_MemProps.memoryTypes[typeIndex].propertyFlags;
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                                json.WriteString("DEVICE_LOCAL");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                                                json.WriteString("HOST_VISIBLE");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                                json.WriteString("HOST_COHERENT");
                                            if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                                                json.WriteString("HOST_CACHED");
                                            if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
                                                json.WriteString("LAZILY_ALLOCATED");
                                        #if VMA_VULKAN_VERSION >= 1001000
                                            if (flags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
                                                json.WriteString("PROTECTED");
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY)
                                                json.WriteString("DEVICE_COHERENT_AMD");
                                            if (flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)
                                                json.WriteString("DEVICE_UNCACHED_AMD");
                                        #endif

                                            flags &= ~(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                        #if VMA_VULKAN_VERSION >= 1001000
                                                | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
                                        #endif
                                        #if VK_AMD_device_coherent_memory
                                                | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY
                                                | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY
                                        #endif
                                                | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                                            if (flags != 0)
                                                json.WriteNumber(flags);
                                        }
                                        json.EndArray();

                                        json.WriteString("Stats");
                                        VmaPrintDetailedStatistics(json, stats.memoryType[typeIndex]);
                                    }
                                    json.EndObject();
                                }
                            }

                        }
                        json.EndObject();
                    }
                    json.EndObject();
                }
            }
            json.EndObject();
        }

        if (detailedMap == VK_TRUE)
            allocator->PrintDetailedMap(json);

        json.EndObject();
    }

    *ppStatsString = VmaCreateStringCopy(allocator->GetAllocationCallbacks(), sb.GetData(), sb.GetLength());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
    VmaAllocator allocator,
    char* pStatsString)
{
    if(pStatsString != VMA_NULL)
    {
        VMA_ASSERT(allocator);
        VmaFreeString(allocator->GetAllocationCallbacks(), pStatsString);
    }
}

#endif // VMA_STATS_STRING_ENABLED

/*
This function is not protected by any mutex because it just reads immutable data.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
    VmaAllocator allocator,
    uint32_t memoryTypeBits,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    return allocator->FindMemoryTypeIndex(memoryTypeBits, pAllocationCreateInfo, UINT32_MAX, pMemoryTypeIndex);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pBufferCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    const VmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
    VkResult res;

#if VMA_VULKAN_VERSION >= 1003000
    if(funcs->vkGetDeviceBufferMemoryRequirements)
    {
        // Can query straight from VkBufferCreateInfo :)
        VkDeviceBufferMemoryRequirements devBufMemReq = {VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS};
        devBufMemReq.pCreateInfo = pBufferCreateInfo;

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceBufferMemoryRequirements)(hDev, &devBufMemReq, &memReq);

        res = allocator->FindMemoryTypeIndex(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo, pBufferCreateInfo->usage, pMemoryTypeIndex);
    }
    else
#endif // #if VMA_VULKAN_VERSION >= 1003000
    {
        // Must create a dummy buffer to query :(
        VkBuffer hBuffer = VK_NULL_HANDLE;
        res = funcs->vkCreateBuffer(
            hDev, pBufferCreateInfo, allocator->GetAllocationCallbacks(), &hBuffer);
        if(res == VK_SUCCESS)
        {
            VkMemoryRequirements memReq = {};
            funcs->vkGetBufferMemoryRequirements(hDev, hBuffer, &memReq);

            res = allocator->FindMemoryTypeIndex(
                memReq.memoryTypeBits, pAllocationCreateInfo, pBufferCreateInfo->usage, pMemoryTypeIndex);

            funcs->vkDestroyBuffer(
                hDev, hBuffer, allocator->GetAllocationCallbacks());
        }
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    uint32_t* pMemoryTypeIndex)
{
    VMA_ASSERT(allocator != VK_NULL_HANDLE);
    VMA_ASSERT(pImageCreateInfo != VMA_NULL);
    VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
    VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

    const VkDevice hDev = allocator->m_hDevice;
    const VmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
    VkResult res;

#if VMA_VULKAN_VERSION >= 1003000
    if(funcs->vkGetDeviceImageMemoryRequirements)
    {
        // Can query straight from VkImageCreateInfo :)
        VkDeviceImageMemoryRequirements devImgMemReq = {VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS};
        devImgMemReq.pCreateInfo = pImageCreateInfo;
        VMA_ASSERT(pImageCreateInfo->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY && (pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
            "Cannot use this VkImageCreateInfo with vmaFindMemoryTypeIndexForImageInfo as I don't know what to pass as VkDeviceImageMemoryRequirements::planeAspect.");

        VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        (*funcs->vkGetDeviceImageMemoryRequirements)(hDev, &devImgMemReq, &memReq);

        res = allocator->FindMemoryTypeIndex(
            memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo, pImageCreateInfo->usage, pMemoryTypeIndex);
    }
    else
#endif // #if VMA_VULKAN_VERSION >= 1003000
    {
        // Must create a dummy image to query :(
        VkImage hImage = VK_NULL_HANDLE;
        res = funcs->vkCreateImage(
            hDev, pImageCreateInfo, allocator->GetAllocationCallbacks(), &hImage);
        if(res == VK_SUCCESS)
        {
            VkMemoryRequirements memReq = {};
            funcs->vkGetImageMemoryRequirements(hDev, hImage, &memReq);

            res = allocator->FindMemoryTypeIndex(
                memReq.memoryTypeBits, pAllocationCreateInfo, pImageCreateInfo->usage, pMemoryTypeIndex);

            funcs->vkDestroyImage(
                hDev, hImage, allocator->GetAllocationCallbacks());
        }
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
    VmaAllocator allocator,
    const VmaPoolCreateInfo* pCreateInfo,
    VmaPool* pPool)
{
    VMA_ASSERT(allocator && pCreateInfo && pPool);

    VMA_DEBUG_LOG("vmaCreatePool");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CreatePool(pCreateInfo, pPool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
    VmaAllocator allocator,
    VmaPool pool)
{
    VMA_ASSERT(allocator);

    if(pool == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyPool");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->DestroyPool(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStatistics(
    VmaAllocator allocator,
    VmaPool pool,
    VmaStatistics* pPoolStats)
{
    VMA_ASSERT(allocator && pool && pPoolStats);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetPoolStatistics(pool, pPoolStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculatePoolStatistics(
    VmaAllocator allocator,
    VmaPool pool,
    VmaDetailedStatistics* pPoolStats)
{
    VMA_ASSERT(allocator && pool && pPoolStats);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->CalculatePoolStatistics(pool, pPoolStats);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(VmaAllocator allocator, VmaPool pool)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VMA_DEBUG_LOG("vmaCheckPoolCorruption");

    return allocator->CheckPoolCorruption(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char** ppName)
{
    VMA_ASSERT(allocator && pool && ppName);

    VMA_DEBUG_LOG("vmaGetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *ppName = pool->GetName();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
    VmaAllocator allocator,
    VmaPool pool,
    const char* pName)
{
    VMA_ASSERT(allocator && pool);

    VMA_DEBUG_LOG("vmaSetPoolName");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    pool->SetName(pName);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        UINT32_MAX, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
    VmaAllocator allocator,
    const VkMemoryRequirements* pVkMemoryRequirements,
    const VmaAllocationCreateInfo* pCreateInfo,
    size_t allocationCount,
    VmaAllocation* pAllocations,
    VmaAllocationInfo* pAllocationInfo)
{
    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocations);

    VMA_DEBUG_LOG("vmaAllocateMemoryPages");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkResult result = allocator->AllocateMemory(
        *pVkMemoryRequirements,
        false, // requiresDedicatedAllocation
        false, // prefersDedicatedAllocation
        VK_NULL_HANDLE, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        UINT32_MAX, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_UNKNOWN,
        allocationCount,
        pAllocations);

    if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
    {
        for(size_t i = 0; i < allocationCount; ++i)
        {
            allocator->GetAllocationInfo(pAllocations[i], pAllocationInfo + i);
        }
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && buffer != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
    allocator->GetBufferMemoryRequirements(buffer, vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        buffer, // dedicatedBuffer
        VK_NULL_HANDLE, // dedicatedImage
        UINT32_MAX, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_BUFFER,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
    VmaAllocator allocator,
    VkImage image,
    const VmaAllocationCreateInfo* pCreateInfo,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && image != VK_NULL_HANDLE && pCreateInfo && pAllocation);

    VMA_DEBUG_LOG("vmaAllocateMemoryForImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    VkMemoryRequirements vkMemReq = {};
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation  = false;
    allocator->GetImageMemoryRequirements(image, vkMemReq,
        requiresDedicatedAllocation, prefersDedicatedAllocation);

    VkResult result = allocator->AllocateMemory(
        vkMemReq,
        requiresDedicatedAllocation,
        prefersDedicatedAllocation,
        VK_NULL_HANDLE, // dedicatedBuffer
        image, // dedicatedImage
        UINT32_MAX, // dedicatedBufferImageUsage
        *pCreateInfo,
        VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN,
        1, // allocationCount
        pAllocation);

    if(pAllocationInfo && result == VK_SUCCESS)
    {
        allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
    }

    return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaFreeMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FreeMemory(
        1, // allocationCount
        &allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
    VmaAllocator allocator,
    size_t allocationCount,
    const VmaAllocation* pAllocations)
{
    if(allocationCount == 0)
    {
        return;
    }

    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaFreeMemoryPages");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->FreeMemory(allocationCount, pAllocations);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && allocation && pAllocationInfo);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->GetAllocationInfo(allocation, pAllocationInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void* pUserData)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocation->SetUserData(allocator, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationName(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const char* VMA_NULLABLE pName)
{
    allocation->SetName(allocator, pName);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationMemoryProperties(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkMemoryPropertyFlags* VMA_NOT_NULL pFlags)
{
    VMA_ASSERT(allocator && allocation && pFlags);
    const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
    *pFlags = allocator->m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    void** ppData)
{
    VMA_ASSERT(allocator && allocation && ppData);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->Map(allocation, ppData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
    VmaAllocator allocator,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    allocator->Unmap(allocation);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaFlushAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    const VkResult res = allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_FLUSH);

    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize offset,
    VkDeviceSize size)
{
    VMA_ASSERT(allocator && allocation);

    VMA_DEBUG_LOG("vmaInvalidateAllocation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    const VkResult res = allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_INVALIDATE);

    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocations(
    VmaAllocator allocator,
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocations);

    VMA_DEBUG_LOG("vmaFlushAllocations");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    const VkResult res = allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_FLUSH);

    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocations(
    VmaAllocator allocator,
    uint32_t allocationCount,
    const VmaAllocation* allocations,
    const VkDeviceSize* offsets,
    const VkDeviceSize* sizes)
{
    VMA_ASSERT(allocator);

    if(allocationCount == 0)
    {
        return VK_SUCCESS;
    }

    VMA_ASSERT(allocations);

    VMA_DEBUG_LOG("vmaInvalidateAllocations");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    const VkResult res = allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_INVALIDATE);

    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(
    VmaAllocator allocator,
    uint32_t memoryTypeBits)
{
    VMA_ASSERT(allocator);

    VMA_DEBUG_LOG("vmaCheckCorruption");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->CheckCorruption(memoryTypeBits);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentation(
    VmaAllocator allocator,
    const VmaDefragmentationInfo* pInfo,
    VmaDefragmentationContext* pContext)
{
    VMA_ASSERT(allocator && pInfo && pContext);

    VMA_DEBUG_LOG("vmaBeginDefragmentation");

    if (pInfo->pool != VMA_NULL)
    {
        // Check if run on supported algorithms
        if (pInfo->pool->m_BlockVector.GetAlgorithm() & VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
            return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pContext = vma_new(allocator, VmaDefragmentationContext_T)(allocator, *pInfo);
    return VK_SUCCESS;
}

VMA_CALL_PRE void VMA_CALL_POST vmaEndDefragmentation(
    VmaAllocator allocator,
    VmaDefragmentationContext context,
    VmaDefragmentationStats* pStats)
{
    VMA_ASSERT(allocator && context);

    VMA_DEBUG_LOG("vmaEndDefragmentation");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if (pStats)
        context->GetStats(*pStats);
    vma_delete(allocator, context);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
    VMA_ASSERT(context && pPassInfo);

    VMA_DEBUG_LOG("vmaBeginDefragmentationPass");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->DefragmentPassBegin(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaEndDefragmentationPass(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaDefragmentationContext VMA_NOT_NULL context,
    VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
    VMA_ASSERT(context && pPassInfo);

    VMA_DEBUG_LOG("vmaEndDefragmentationPass");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return context->DefragmentPassEnd(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkBuffer buffer)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, 0, buffer, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkBuffer buffer,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && buffer);

    VMA_DEBUG_LOG("vmaBindBufferMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindBufferMemory(allocation, allocationLocalOffset, buffer, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkImage image)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    return allocator->BindImageMemory(allocation, 0, image, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
    VmaAllocator allocator,
    VmaAllocation allocation,
    VkDeviceSize allocationLocalOffset,
    VkImage image,
    const void* pNext)
{
    VMA_ASSERT(allocator && allocation && image);

    VMA_DEBUG_LOG("vmaBindImageMemory2");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

        return allocator->BindImageMemory(allocation, allocationLocalOffset, image, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);

    if(pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->m_UseKhrBufferDeviceAddress)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkBuffer.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
        allocator->m_hDevice,
        pBufferCreateInfo,
        allocator->GetAllocationCallbacks(),
        pBuffer);
    if(res >= 0)
    {
        // 2. vkGetBufferMemoryRequirements.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->GetBufferMemoryRequirements(*pBuffer, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        // 3. Allocate memory using allocator.
        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pBuffer, // dedicatedBuffer
            VK_NULL_HANDLE, // dedicatedImage
            pBufferCreateInfo->usage, // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);

        if(res >= 0)
        {
            // 3. Bind buffer with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindBufferMemory(*pAllocation, 0, *pBuffer, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->InitBufferImageUsage(pBufferCreateInfo->usage);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
            *pBuffer = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
        *pBuffer = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBufferWithAlignment(
    VmaAllocator allocator,
    const VkBufferCreateInfo* pBufferCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkDeviceSize minAlignment,
    VkBuffer* pBuffer,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && VmaIsPow2(minAlignment) && pBuffer && pAllocation);

    if(pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->m_UseKhrBufferDeviceAddress)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateBufferWithAlignment");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pBuffer = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkBuffer.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
        allocator->m_hDevice,
        pBufferCreateInfo,
        allocator->GetAllocationCallbacks(),
        pBuffer);
    if(res >= 0)
    {
        // 2. vkGetBufferMemoryRequirements.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->GetBufferMemoryRequirements(*pBuffer, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        // 2a. Include minAlignment
        vkMemReq.alignment = VMA_MAX(vkMemReq.alignment, minAlignment);

        // 3. Allocate memory using allocator.
        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            *pBuffer, // dedicatedBuffer
            VK_NULL_HANDLE, // dedicatedImage
            pBufferCreateInfo->usage, // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            VMA_SUBALLOCATION_TYPE_BUFFER,
            1, // allocationCount
            pAllocation);

        if(res >= 0)
        {
            // 3. Bind buffer with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindBufferMemory(*pAllocation, 0, *pBuffer, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->InitBufferImageUsage(pBufferCreateInfo->usage);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
            *pBuffer = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
        *pBuffer = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
    return vmaCreateAliasingBuffer2(allocator, allocation, 0, pBufferCreateInfo, pBuffer);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
    VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
    VMA_ASSERT(allocator && pBufferCreateInfo && pBuffer && allocation);
    VMA_ASSERT(allocationLocalOffset + pBufferCreateInfo->size <= allocation->GetSize());

    VMA_DEBUG_LOG("vmaCreateAliasingBuffer2");

    *pBuffer = VK_NULL_HANDLE;

    if (pBufferCreateInfo->size == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
        !allocator->m_UseKhrBufferDeviceAddress)
    {
        VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkBuffer.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
        allocator->m_hDevice,
        pBufferCreateInfo,
        allocator->GetAllocationCallbacks(),
        pBuffer);
    if (res >= 0)
    {
        // 2. Bind buffer with memory.
        res = allocator->BindBufferMemory(allocation, allocationLocalOffset, *pBuffer, VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
    VmaAllocator allocator,
    VkBuffer buffer,
    VmaAllocation allocation)
{
    VMA_ASSERT(allocator);

    if(buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyBuffer");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(buffer != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, buffer, allocator->GetAllocationCallbacks());
    }

    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo)
{
    VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);

    if(pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG("vmaCreateImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    *pImage = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkImage.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
        allocator->m_hDevice,
        pImageCreateInfo,
        allocator->GetAllocationCallbacks(),
        pImage);
    if(res >= 0)
    {
        VmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
            VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
            VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;

        // 2. Allocate memory using allocator.
        VkMemoryRequirements vkMemReq = {};
        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation  = false;
        allocator->GetImageMemoryRequirements(*pImage, vkMemReq,
            requiresDedicatedAllocation, prefersDedicatedAllocation);

        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            VK_NULL_HANDLE, // dedicatedBuffer
            *pImage, // dedicatedImage
            pImageCreateInfo->usage, // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            suballocType,
            1, // allocationCount
            pAllocation);

        if(res >= 0)
        {
            // 3. Bind image with memory.
            if((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindImageMemory(*pAllocation, 0, *pImage, VMA_NULL);
            }
            if(res >= 0)
            {
                // All steps succeeded.
                #if VMA_STATS_STRING_ENABLED
                    (*pAllocation)->InitBufferImageUsage(pImageCreateInfo->usage);
                #endif
                if(pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation);
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
            *pImage = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
        *pImage = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage)
{
    return vmaCreateAliasingImage2(allocator, allocation, 0, pImageCreateInfo, pImage);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage2(
    VmaAllocator VMA_NOT_NULL allocator,
    VmaAllocation VMA_NOT_NULL allocation,
    VkDeviceSize allocationLocalOffset,
    const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage)
{
    VMA_ASSERT(allocator && pImageCreateInfo && pImage && allocation);

    *pImage = VK_NULL_HANDLE;

    VMA_DEBUG_LOG("vmaCreateImage2");

    if (pImageCreateInfo->extent.width == 0 ||
        pImageCreateInfo->extent.height == 0 ||
        pImageCreateInfo->extent.depth == 0 ||
        pImageCreateInfo->mipLevels == 0 ||
        pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    // 1. Create VkImage.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
        allocator->m_hDevice,
        pImageCreateInfo,
        allocator->GetAllocationCallbacks(),
        pImage);
    if (res >= 0)
    {
        // 2. Bind image with memory.
        res = allocator->BindImageMemory(allocation, allocationLocalOffset, *pImage, VMA_NULL);
        if (res >= 0)
        {
            return VK_SUCCESS;
        }
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
    VmaAllocator VMA_NOT_NULL allocator,
    VkImage VMA_NULLABLE_NON_DISPATCHABLE image,
    VmaAllocation VMA_NULLABLE allocation)
{
    VMA_ASSERT(allocator);

    if(image == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
    {
        return;
    }

    VMA_DEBUG_LOG("vmaDestroyImage");

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

    if(image != VK_NULL_HANDLE)
    {
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, image, allocator->GetAllocationCallbacks());
    }
    if(allocation != VK_NULL_HANDLE)
    {
        allocator->FreeMemory(
            1, // allocationCount
            &allocation);
    }
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateVirtualBlock(
    const VmaVirtualBlockCreateInfo* VMA_NOT_NULL pCreateInfo,
    VmaVirtualBlock VMA_NULLABLE * VMA_NOT_NULL pVirtualBlock)
{
    VMA_ASSERT(pCreateInfo && pVirtualBlock);
    VMA_ASSERT(pCreateInfo->size > 0);
    VMA_DEBUG_LOG("vmaCreateVirtualBlock");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    *pVirtualBlock = vma_new(pCreateInfo->pAllocationCallbacks, VmaVirtualBlock_T)(*pCreateInfo);
    VkResult res = (*pVirtualBlock)->Init();
    if(res < 0)
    {
        vma_delete(pCreateInfo->pAllocationCallbacks, *pVirtualBlock);
        *pVirtualBlock = VK_NULL_HANDLE;
    }
    return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyVirtualBlock(VmaVirtualBlock VMA_NULLABLE virtualBlock)
{
    if(virtualBlock != VK_NULL_HANDLE)
    {
        VMA_DEBUG_LOG("vmaDestroyVirtualBlock");
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        VkAllocationCallbacks allocationCallbacks = virtualBlock->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
        vma_delete(&allocationCallbacks, virtualBlock);
    }
}

VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaIsVirtualBlockEmpty(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaIsVirtualBlockEmpty");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->IsEmpty() ? VK_TRUE : VK_FALSE;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualAllocationInfo(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, VmaVirtualAllocationInfo* VMA_NOT_NULL pVirtualAllocInfo)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pVirtualAllocInfo != VMA_NULL);
    VMA_DEBUG_LOG("vmaGetVirtualAllocationInfo");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->GetAllocationInfo(allocation, *pVirtualAllocInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaVirtualAllocate(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    const VmaVirtualAllocationCreateInfo* VMA_NOT_NULL pCreateInfo, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pAllocation,
    VkDeviceSize* VMA_NULLABLE pOffset)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pCreateInfo != VMA_NULL && pAllocation != VMA_NULL);
    VMA_DEBUG_LOG("vmaVirtualAllocate");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    return virtualBlock->Allocate(*pCreateInfo, *pAllocation, pOffset);
}

VMA_CALL_PRE void VMA_CALL_POST vmaVirtualFree(VmaVirtualBlock VMA_NOT_NULL virtualBlock, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE allocation)
{
    if(allocation != VK_NULL_HANDLE)
    {
        VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        VMA_DEBUG_LOG("vmaVirtualFree");
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        virtualBlock->Free(allocation);
    }
}

VMA_CALL_PRE void VMA_CALL_POST vmaClearVirtualBlock(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaClearVirtualBlock");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->Clear();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetVirtualAllocationUserData(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, void* VMA_NULLABLE pUserData)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
    VMA_DEBUG_LOG("vmaSetVirtualAllocationUserData");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->SetAllocationUserData(allocation, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaStatistics* VMA_NOT_NULL pStats)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
    VMA_DEBUG_LOG("vmaGetVirtualBlockStatistics");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->GetStatistics(*pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    VmaDetailedStatistics* VMA_NOT_NULL pStats)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
    VMA_DEBUG_LOG("vmaCalculateVirtualBlockStatistics");
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    virtualBlock->CalculateDetailedStatistics(*pStats);
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE * VMA_NOT_NULL ppStatsString, VkBool32 detailedMap)
{
    VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && ppStatsString != VMA_NULL);
    VMA_DEBUG_GLOBAL_MUTEX_LOCK;
    const VkAllocationCallbacks* allocationCallbacks = virtualBlock->GetAllocationCallbacks();
    VmaStringBuilder sb(allocationCallbacks);
    virtualBlock->BuildStatsString(detailedMap != VK_FALSE, sb);
    *ppStatsString = VmaCreateStringCopy(allocationCallbacks, sb.GetData(), sb.GetLength());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
    char* VMA_NULLABLE pStatsString)
{
    if(pStatsString != VMA_NULL)
    {
        VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
        VMA_DEBUG_GLOBAL_MUTEX_LOCK;
        VmaFreeString(virtualBlock->GetAllocationCallbacks(), pStatsString);
    }
}
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_PUBLIC_INTERFACE
#endif // VMA_IMPLEMENTATION

/**
\page quick_start Quick start

\section quick_start_project_setup Project setup

Vulkan Memory Allocator comes in form of a "stb-style" single header file.
You don't need to build it as a separate library project.
You can add this file directly to your project and submit it to code repository next to your other source files.

"Single header" doesn't mean that everything is contained in C/C++ declarations,
like it tends to be in case of inline functions or C++ templates.
It means that implementation is bundled with interface in a single file and needs to be extracted using preprocessor macro.
If you don't do it properly, you will get linker errors.

To do it properly:

-# Include "vk_mem_alloc.h" file in each CPP file where you want to use the library.
   This includes declarations of all members of the library.
-# In exactly one CPP file define following macro before this include.
   It enables also internal definitions.

\code
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
\endcode

It may be a good idea to create dedicated CPP file just for this purpose.

This library includes header `<vulkan/vulkan.h>`, which in turn
includes `<windows.h>` on Windows. If you need some specific macros defined
before including these headers (like `WIN32_LEAN_AND_MEAN` or
`WINVER` for Windows, `VK_USE_PLATFORM_WIN32_KHR` for Vulkan), you must define
them before every `#include` of this library.

This library is written in C++, but has C-compatible interface.
Thus you can include and use vk_mem_alloc.h in C or C++ code, but full
implementation with `VMA_IMPLEMENTATION` macro must be compiled as C++, NOT as C.
Some features of C++14 are used. STL containers, RTTI, or C++ exceptions are not used.


\section quick_start_initialization Initialization

At program startup:

-# Initialize Vulkan to have `VkPhysicalDevice`, `VkDevice` and `VkInstance` object.
-# Fill VmaAllocatorCreateInfo structure and create #VmaAllocator object by
   calling vmaCreateAllocator().

Only members `physicalDevice`, `device`, `instance` are required.
However, you should inform the library which Vulkan version do you use by setting
VmaAllocatorCreateInfo::vulkanApiVersion and which extensions did you enable
by setting VmaAllocatorCreateInfo::flags (like #VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT for VK_KHR_buffer_device_address).
Otherwise, VMA would use only features of Vulkan 1.0 core with no extensions.

\subsection quick_start_initialization_selecting_vulkan_version Selecting Vulkan version

VMA supports Vulkan version down to 1.0, for backward compatibility.
If you want to use higher version, you need to inform the library about it.
This is a two-step process.

<b>Step 1: Compile time.</b> By default, VMA compiles with code supporting the highest
Vulkan version found in the included `<vulkan/vulkan.h>` that is also supported by the library.
If this is OK, you don't need to do anything.
However, if you want to compile VMA as if only some lower Vulkan version was available,
define macro `VMA_VULKAN_VERSION` before every `#include "vk_mem_alloc.h"`.
It should have decimal numeric value in form of ABBBCCC, where A = major, BBB = minor, CCC = patch Vulkan version.
For example, to compile against Vulkan 1.2:

\code
#define VMA_VULKAN_VERSION 1002000 // Vulkan 1.2
#include "vk_mem_alloc.h"
\endcode

<b>Step 2: Runtime.</b> Even when compiled with higher Vulkan version available,
VMA can use only features of a lower version, which is configurable during creation of the #VmaAllocator object.
By default, only Vulkan 1.0 is used.
To initialize the allocator with support for higher Vulkan version, you need to set member
VmaAllocatorCreateInfo::vulkanApiVersion to an appropriate value, e.g. using constants like `VK_API_VERSION_1_2`.
See code sample below.

\subsection quick_start_initialization_importing_vulkan_functions Importing Vulkan functions

You may need to configure importing Vulkan functions. There are 3 ways to do this:

-# **If you link with Vulkan static library** (e.g. "vulkan-1.lib" on Windows):
   - You don't need to do anything.
   - VMA will use these, as macro `VMA_STATIC_VULKAN_FUNCTIONS` is defined to 1 by default.
-# **If you want VMA to fetch pointers to Vulkan functions dynamically** using `vkGetInstanceProcAddr`,
   `vkGetDeviceProcAddr` (this is the option presented in the example below):
   - Define `VMA_STATIC_VULKAN_FUNCTIONS` to 0, `VMA_DYNAMIC_VULKAN_FUNCTIONS` to 1.
   - Provide pointers to these two functions via VmaVulkanFunctions::vkGetInstanceProcAddr,
     VmaVulkanFunctions::vkGetDeviceProcAddr.
   - The library will fetch pointers to all other functions it needs internally.
-# **If you fetch pointers to all Vulkan functions in a custom way**, e.g. using some loader like
   [Volk](https://github.com/zeux/volk):
   - Define `VMA_STATIC_VULKAN_FUNCTIONS` and `VMA_DYNAMIC_VULKAN_FUNCTIONS` to 0.
   - Pass these pointers via structure #VmaVulkanFunctions.

Example for case 2:

\code
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_mem_alloc.h"

...

VmaVulkanFunctions vulkanFunctions = {};
vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

VmaAllocatorCreateInfo allocatorCreateInfo = {};
allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
allocatorCreateInfo.physicalDevice = physicalDevice;
allocatorCreateInfo.device = device;
allocatorCreateInfo.instance = instance;
allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

VmaAllocator allocator;
vmaCreateAllocator(&allocatorCreateInfo, &allocator);
\endcode


\section quick_start_resource_allocation Resource allocation

When you want to create a buffer or image:

-# Fill `VkBufferCreateInfo` / `VkImageCreateInfo` structure.
-# Fill VmaAllocationCreateInfo structure.
-# Call vmaCreateBuffer() / vmaCreateImage() to get `VkBuffer`/`VkImage` with memory
   already allocated and bound to it, plus #VmaAllocation objects that represents its underlying memory.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

Don't forget to destroy your objects when no longer needed:

\code
vmaDestroyBuffer(allocator, buffer, allocation);
vmaDestroyAllocator(allocator);
\endcode


\page choosing_memory_type Choosing memory type

Physical devices in Vulkan support various combinations of memory heaps and
types. Help with choosing correct and optimal memory type for your specific
resource is one of the key features of this library. You can use it by filling
appropriate members of VmaAllocationCreateInfo structure, as described below.
You can also combine multiple methods.

-# If you just want to find memory type index that meets your requirements, you
   can use function: vmaFindMemoryTypeIndexForBufferInfo(),
   vmaFindMemoryTypeIndexForImageInfo(), vmaFindMemoryTypeIndex().
-# If you want to allocate a region of device memory without association with any
   specific image or buffer, you can use function vmaAllocateMemory(). Usage of
   this function is not recommended and usually not needed.
   vmaAllocateMemoryPages() function is also provided for creating multiple allocations at once,
   which may be useful for sparse binding.
-# If you already have a buffer or an image created, you want to allocate memory
   for it and then you will bind it yourself, you can use function
   vmaAllocateMemoryForBuffer(), vmaAllocateMemoryForImage().
   For binding you should use functions: vmaBindBufferMemory(), vmaBindImageMemory()
   or their extended versions: vmaBindBufferMemory2(), vmaBindImageMemory2().
-# **This is the easiest and recommended way to use this library:**
   If you want to create a buffer or an image, allocate memory for it and bind
   them together, all in one call, you can use function vmaCreateBuffer(),
   vmaCreateImage().

When using 3. or 4., the library internally queries Vulkan for memory types
supported for that buffer or image (function `vkGetBufferMemoryRequirements()`)
and uses only one of these types.

If no memory type can be found that meets all the requirements, these functions
return `VK_ERROR_FEATURE_NOT_PRESENT`.

You can leave VmaAllocationCreateInfo structure completely filled with zeros.
It means no requirements are specified for memory type.
It is valid, although not very useful.

\section choosing_memory_type_usage Usage

The easiest way to specify memory requirements is to fill member
VmaAllocationCreateInfo::usage using one of the values of enum #VmaMemoryUsage.
It defines high level, common usage types.
Since version 3 of the library, it is recommended to use #VMA_MEMORY_USAGE_AUTO to let it select best memory type for your resource automatically.

For example, if you want to create a uniform buffer that will be filled using
transfer only once or infrequently and then used for rendering every frame as a uniform buffer, you can
do it using following code. The buffer will most likely end up in a memory type with
`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` to be fast to access by the GPU device.

\code
VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufferInfo.size = 65536;
bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

If you have a preference for putting the resource in GPU (device) memory or CPU (host) memory
on systems with discrete graphics card that have the memories separate, you can use
#VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE or #VMA_MEMORY_USAGE_AUTO_PREFER_HOST.

When using `VMA_MEMORY_USAGE_AUTO*` while you want to map the allocated memory,
you also need to specify one of the host access flags:
#VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
This will help the library decide about preferred memory type to ensure it has `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
so you can map it.

For example, a staging buffer that will be filled via mapped pointer and then
used as a source of transfer to the buffer described previously can be created like this.
It will likely end up in a memory type that is `HOST_VISIBLE` and `HOST_COHERENT`
but not `HOST_CACHED` (meaning uncached, write-combined) and not `DEVICE_LOCAL` (meaning system RAM).

\code
VkBufferCreateInfo stagingBufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
stagingBufferInfo.size = 65536;
stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo stagingAllocInfo = {};
stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

VkBuffer stagingBuffer;
VmaAllocation stagingAllocation;
vmaCreateBuffer(allocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);
\endcode

For more examples of creating different kinds of resources, see chapter \ref usage_patterns.

Usage values `VMA_MEMORY_USAGE_AUTO*` are legal to use only when the library knows
about the resource being created by having `VkBufferCreateInfo` / `VkImageCreateInfo` passed,
so they work with functions like: vmaCreateBuffer(), vmaCreateImage(), vmaFindMemoryTypeIndexForBufferInfo() etc.
If you allocate raw memory using function vmaAllocateMemory(), you have to use other means of selecting
memory type, as described below.

\note
Old usage values (`VMA_MEMORY_USAGE_GPU_ONLY`, `VMA_MEMORY_USAGE_CPU_ONLY`,
`VMA_MEMORY_USAGE_CPU_TO_GPU`, `VMA_MEMORY_USAGE_GPU_TO_CPU`, `VMA_MEMORY_USAGE_CPU_COPY`)
are still available and work same way as in previous versions of the library
for backward compatibility, but they are not recommended.

\section choosing_memory_type_required_preferred_flags Required and preferred flags

You can specify more detailed requirements by filling members
VmaAllocationCreateInfo::requiredFlags and VmaAllocationCreateInfo::preferredFlags
with a combination of bits from enum `VkMemoryPropertyFlags`. For example,
if you want to create a buffer that will be persistently mapped on host (so it
must be `HOST_VISIBLE`) and preferably will also be `HOST_COHERENT` and `HOST_CACHED`,
use following code:

\code
VmaAllocationCreateInfo allocInfo = {};
allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
allocInfo.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode

A memory type is chosen that has all the required flags and as many preferred
flags set as possible.

Value passed in VmaAllocationCreateInfo::usage is internally converted to a set of required and preferred flags,
plus some extra "magic" (heuristics).

\section choosing_memory_type_explicit_memory_types Explicit memory types

If you inspected memory types available on the physical device and you have
a preference for memory types that you want to use, you can fill member
VmaAllocationCreateInfo::memoryTypeBits. It is a bit mask, where each bit set
means that a memory type with that index is allowed to be used for the
allocation. Special value 0, just like `UINT32_MAX`, means there are no
restrictions to memory type index.

Please note that this member is NOT just a memory type index.
Still you can use it to choose just one, specific memory type.
For example, if you already determined that your buffer should be created in
memory type 2, use following code:

\code
uint32_t memoryTypeIndex = 2;

VmaAllocationCreateInfo allocInfo = {};
allocInfo.memoryTypeBits = 1u << memoryTypeIndex;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
\endcode


\section choosing_memory_type_custom_memory_pools Custom memory pools

If you allocate from custom memory pool, all the ways of specifying memory
requirements described above are not applicable and the aforementioned members
of VmaAllocationCreateInfo structure are ignored. Memory type is selected
explicitly when creating the pool and then used to make all the allocations from
that pool. For further details, see \ref custom_memory_pools.

\section choosing_memory_type_dedicated_allocations Dedicated allocations

Memory for allocations is reserved out of larger block of `VkDeviceMemory`
allocated from Vulkan internally. That is the main feature of this whole library.
You can still request a separate memory block to be created for an allocation,
just like you would do in a trivial solution without using any allocator.
In that case, a buffer or image is always bound to that memory at offset 0.
This is called a "dedicated allocation".
You can explicitly request it by using flag #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
The library can also internally decide to use dedicated allocation in some cases, e.g.:

- When the size of the allocation is large.
- When [VK_KHR_dedicated_allocation](@ref vk_khr_dedicated_allocation) extension is enabled
  and it reports that dedicated allocation is required or recommended for the resource.
- When allocation of next big memory block fails due to not enough device memory,
  but allocation with the exact requested size succeeds.


\page memory_mapping Memory mapping

To "map memory" in Vulkan means to obtain a CPU pointer to `VkDeviceMemory`,
to be able to read from it or write to it in CPU code.
Mapping is possible only of memory allocated from a memory type that has
`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` flag.
Functions `vkMapMemory()`, `vkUnmapMemory()` are designed for this purpose.
You can use them directly with memory allocated by this library,
but it is not recommended because of following issue:
Mapping the same `VkDeviceMemory` block multiple times is illegal - only one mapping at a time is allowed.
This includes mapping disjoint regions. Mapping is not reference-counted internally by Vulkan.
Because of this, Vulkan Memory Allocator provides following facilities:

\note If you want to be able to map an allocation, you need to specify one of the flags
#VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
in VmaAllocationCreateInfo::flags. These flags are required for an allocation to be mappable
when using #VMA_MEMORY_USAGE_AUTO or other `VMA_MEMORY_USAGE_AUTO*` enum values.
For other usage values they are ignored and every such allocation made in `HOST_VISIBLE` memory type is mappable,
but they can still be used for consistency.

\section memory_mapping_mapping_functions Mapping functions

The library provides following functions for mapping of a specific #VmaAllocation: vmaMapMemory(), vmaUnmapMemory().
They are safer and more convenient to use than standard Vulkan functions.
You can map an allocation multiple times simultaneously - mapping is reference-counted internally.
You can also map different allocations simultaneously regardless of whether they use the same `VkDeviceMemory` block.
The way it is implemented is that the library always maps entire memory block, not just region of the allocation.
For further details, see description of vmaMapMemory() function.
Example:

\code
// Having these objects initialized:
struct ConstantBuffer
{
    ...
};
ConstantBuffer constantBufferData = ...

VmaAllocator allocator = ...
VkBuffer constantBuffer = ...
VmaAllocation constantBufferAllocation = ...

// You can map and fill your buffer using following code:

void* mappedData;
vmaMapMemory(allocator, constantBufferAllocation, &mappedData);
memcpy(mappedData, &constantBufferData, sizeof(constantBufferData));
vmaUnmapMemory(allocator, constantBufferAllocation);
\endcode

When mapping, you may see a warning from Vulkan validation layer similar to this one:

<i>Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.</i>

It happens because the library maps entire `VkDeviceMemory` block, where different
types of images and buffers may end up together, especially on GPUs with unified memory like Intel.
You can safely ignore it if you are sure you access only memory of the intended
object that you wanted to map.


\section memory_mapping_persistently_mapped_memory Persistently mapped memory

Keeping your memory persistently mapped is generally OK in Vulkan.
You don't need to unmap it before using its data on the GPU.
The library provides a special feature designed for that:
Allocations made with #VMA_ALLOCATION_CREATE_MAPPED_BIT flag set in
VmaAllocationCreateInfo::flags stay mapped all the time,
so you can just access CPU pointer to it any time
without a need to call any "map" or "unmap" function.
Example:

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = sizeof(ConstantBuffer);
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

// Buffer is already mapped. You can access its memory.
memcpy(allocInfo.pMappedData, &constantBufferData, sizeof(constantBufferData));
\endcode

\note #VMA_ALLOCATION_CREATE_MAPPED_BIT by itself doesn't guarantee that the allocation will end up
in a mappable memory type.
For this, you need to also specify #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT or
#VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
#VMA_ALLOCATION_CREATE_MAPPED_BIT only guarantees that if the memory is `HOST_VISIBLE`, the allocation will be mapped on creation.
For an example of how to make use of this fact, see section \ref usage_patterns_advanced_data_uploading.

\section memory_mapping_cache_control Cache flush and invalidate

Memory in Vulkan doesn't need to be unmapped before using it on GPU,
but unless a memory types has `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` flag set,
you need to manually **invalidate** cache before reading of mapped pointer
and **flush** cache after writing to mapped pointer.
Map/unmap operations don't do that automatically.
Vulkan provides following functions for this purpose `vkFlushMappedMemoryRanges()`,
`vkInvalidateMappedMemoryRanges()`, but this library provides more convenient
functions that refer to given allocation object: vmaFlushAllocation(),
vmaInvalidateAllocation(),
or multiple objects at once: vmaFlushAllocations(), vmaInvalidateAllocations().

Regions of memory specified for flush/invalidate must be aligned to
`VkPhysicalDeviceLimits::nonCoherentAtomSize`. This is automatically ensured by the library.
In any memory type that is `HOST_VISIBLE` but not `HOST_COHERENT`, all allocations
within blocks are aligned to this value, so their offsets are always multiply of
`nonCoherentAtomSize` and two different allocations never share same "line" of this size.

Also, Windows drivers from all 3 PC GPU vendors (AMD, Intel, NVIDIA)
currently provide `HOST_COHERENT` flag on all memory types that are
`HOST_VISIBLE`, so on PC you may not need to bother.


\page staying_within_budget Staying within budget

When developing a graphics-intensive game or program, it is important to avoid allocating
more GPU memory than it is physically available. When the memory is over-committed,
various bad things can happen, depending on the specific GPU, graphics driver, and
operating system:

- It may just work without any problems.
- The application may slow down because some memory blocks are moved to system RAM
  and the GPU has to access them through PCI Express bus.
- A new allocation may take very long time to complete, even few seconds, and possibly
  freeze entire system.
- The new allocation may fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
- It may even result in GPU crash (TDR), observed as `VK_ERROR_DEVICE_LOST`
  returned somewhere later.

\section staying_within_budget_querying_for_budget Querying for budget

To query for current memory usage and available budget, use function vmaGetHeapBudgets().
Returned structure #VmaBudget contains quantities expressed in bytes, per Vulkan memory heap.

Please note that this function returns different information and works faster than
vmaCalculateStatistics(). vmaGetHeapBudgets() can be called every frame or even before every
allocation, while vmaCalculateStatistics() is intended to be used rarely,
only to obtain statistical information, e.g. for debugging purposes.

It is recommended to use <b>VK_EXT_memory_budget</b> device extension to obtain information
about the budget from Vulkan device. VMA is able to use this extension automatically.
When not enabled, the allocator behaves same way, but then it estimates current usage
and available budget based on its internal information and Vulkan memory heap sizes,
which may be less precise. In order to use this extension:

1. Make sure extensions VK_EXT_memory_budget and VK_KHR_get_physical_device_properties2
   required by it are available and enable them. Please note that the first is a device
   extension and the second is instance extension!
2. Use flag #VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT when creating #VmaAllocator object.
3. Make sure to call vmaSetCurrentFrameIndex() every frame. Budget is queried from
   Vulkan inside of it to avoid overhead of querying it with every allocation.

\section staying_within_budget_controlling_memory_usage Controlling memory usage

There are many ways in which you can try to stay within the budget.

First, when making new allocation requires allocating a new memory block, the library
tries not to exceed the budget automatically. If a block with default recommended size
(e.g. 256 MB) would go over budget, a smaller block is allocated, possibly even
dedicated memory for just this resource.

If the size of the requested resource plus current memory usage is more than the
budget, by default the library still tries to create it, leaving it to the Vulkan
implementation whether the allocation succeeds or fails. You can change this behavior
by using #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag. With it, the allocation is
not made if it would exceed the budget or if the budget is already exceeded.
VMA then tries to make the allocation from the next eligible Vulkan memory type.
The all of them fail, the call then fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
Example usage pattern may be to pass the #VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT flag
when creating resources that are not essential for the application (e.g. the texture
of a specific object) and not to pass it when creating critically important resources
(e.g. render targets).

On AMD graphics cards there is a custom vendor extension available: <b>VK_AMD_memory_overallocation_behavior</b>
that allows to control the behavior of the Vulkan implementation in out-of-memory cases -
whether it should fail with an error code or still allow the allocation.
Usage of this extension involves only passing extra structure on Vulkan device creation,
so it is out of scope of this library.

Finally, you can also use #VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT flag to make sure
a new allocation is created only when it fits inside one of the existing memory blocks.
If it would require to allocate a new block, if fails instead with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.
This also ensures that the function call is very fast because it never goes to Vulkan
to obtain a new block.

\note Creating \ref custom_memory_pools with VmaPoolCreateInfo::minBlockCount
set to more than 0 will currently try to allocate memory blocks without checking whether they
fit within budget.


\page resource_aliasing Resource aliasing (overlap)

New explicit graphics APIs (Vulkan and Direct3D 12), thanks to manual memory
management, give an opportunity to alias (overlap) multiple resources in the
same region of memory - a feature not available in the old APIs (Direct3D 11, OpenGL).
It can be useful to save video memory, but it must be used with caution.

For example, if you know the flow of your whole render frame in advance, you
are going to use some intermediate textures or buffers only during a small range of render passes,
and you know these ranges don't overlap in time, you can bind these resources to
the same place in memory, even if they have completely different parameters (width, height, format etc.).

![Resource aliasing (overlap)](../gfx/Aliasing.png)

Such scenario is possible using VMA, but you need to create your images manually.
Then you need to calculate parameters of an allocation to be made using formula:

- allocation size = max(size of each image)
- allocation alignment = max(alignment of each image)
- allocation memoryTypeBits = bitwise AND(memoryTypeBits of each image)

Following example shows two different images bound to the same place in memory,
allocated to fit largest of them.

\code
// A 512x512 texture to be sampled.
VkImageCreateInfo img1CreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
img1CreateInfo.imageType = VK_IMAGE_TYPE_2D;
img1CreateInfo.extent.width = 512;
img1CreateInfo.extent.height = 512;
img1CreateInfo.extent.depth = 1;
img1CreateInfo.mipLevels = 10;
img1CreateInfo.arrayLayers = 1;
img1CreateInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
img1CreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
img1CreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
img1CreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
img1CreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

// A full screen texture to be used as color attachment.
VkImageCreateInfo img2CreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
img2CreateInfo.imageType = VK_IMAGE_TYPE_2D;
img2CreateInfo.extent.width = 1920;
img2CreateInfo.extent.height = 1080;
img2CreateInfo.extent.depth = 1;
img2CreateInfo.mipLevels = 1;
img2CreateInfo.arrayLayers = 1;
img2CreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
img2CreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
img2CreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
img2CreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
img2CreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VkImage img1;
res = vkCreateImage(device, &img1CreateInfo, nullptr, &img1);
VkImage img2;
res = vkCreateImage(device, &img2CreateInfo, nullptr, &img2);

VkMemoryRequirements img1MemReq;
vkGetImageMemoryRequirements(device, img1, &img1MemReq);
VkMemoryRequirements img2MemReq;
vkGetImageMemoryRequirements(device, img2, &img2MemReq);

VkMemoryRequirements finalMemReq = {};
finalMemReq.size = std::max(img1MemReq.size, img2MemReq.size);
finalMemReq.alignment = std::max(img1MemReq.alignment, img2MemReq.alignment);
finalMemReq.memoryTypeBits = img1MemReq.memoryTypeBits & img2MemReq.memoryTypeBits;
// Validate if(finalMemReq.memoryTypeBits != 0)

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

VmaAllocation alloc;
res = vmaAllocateMemory(allocator, &finalMemReq, &allocCreateInfo, &alloc, nullptr);

res = vmaBindImageMemory(allocator, alloc, img1);
res = vmaBindImageMemory(allocator, alloc, img2);

// You can use img1, img2 here, but not at the same time!

vmaFreeMemory(allocator, alloc);
vkDestroyImage(allocator, img2, nullptr);
vkDestroyImage(allocator, img1, nullptr);
\endcode

VMA also provides convenience functions that create a buffer or image and bind it to memory
represented by an existing #VmaAllocation:
vmaCreateAliasingBuffer(), vmaCreateAliasingBuffer2(),
vmaCreateAliasingImage(), vmaCreateAliasingImage2().
Versions with "2" offer additional parameter `allocationLocalOffset`.

Remember that using resources that alias in memory requires proper synchronization.
You need to issue a memory barrier to make sure commands that use `img1` and `img2`
don't overlap on GPU timeline.
You also need to treat a resource after aliasing as uninitialized - containing garbage data.
For example, if you use `img1` and then want to use `img2`, you need to issue
an image memory barrier for `img2` with `oldLayout` = `VK_IMAGE_LAYOUT_UNDEFINED`.

Additional considerations:

- Vulkan also allows to interpret contents of memory between aliasing resources consistently in some cases.
See chapter 11.8. "Memory Aliasing" of Vulkan specification or `VK_IMAGE_CREATE_ALIAS_BIT` flag.
- You can create more complex layout where different images and buffers are bound
at different offsets inside one large allocation. For example, one can imagine
a big texture used in some render passes, aliasing with a set of many small buffers
used between in some further passes. To bind a resource at non-zero offset in an allocation,
use vmaBindBufferMemory2() / vmaBindImageMemory2().
- Before allocating memory for the resources you want to alias, check `memoryTypeBits`
returned in memory requirements of each resource to make sure the bits overlap.
Some GPUs may expose multiple memory types suitable e.g. only for buffers or
images with `COLOR_ATTACHMENT` usage, so the sets of memory types supported by your
resources may be disjoint. Aliasing them is not possible in that case.


\page custom_memory_pools Custom memory pools

A memory pool contains a number of `VkDeviceMemory` blocks.
The library automatically creates and manages default pool for each memory type available on the device.
Default memory pool automatically grows in size.
Size of allocated blocks is also variable and managed automatically.

You can create custom pool and allocate memory out of it.
It can be useful if you want to:

- Keep certain kind of allocations separate from others.
- Enforce particular, fixed size of Vulkan memory blocks.
- Limit maximum amount of Vulkan memory allocated for that pool.
- Reserve minimum or fixed amount of Vulkan memory always preallocated for that pool.
- Use extra parameters for a set of your allocations that are available in #VmaPoolCreateInfo but not in
  #VmaAllocationCreateInfo - e.g., custom minimum alignment, custom `pNext` chain.
- Perform defragmentation on a specific subset of your allocations.

To use custom memory pools:

-# Fill VmaPoolCreateInfo structure.
-# Call vmaCreatePool() to obtain #VmaPool handle.
-# When making an allocation, set VmaAllocationCreateInfo::pool to this handle.
   You don't need to specify any other parameters of this structure, like `usage`.

Example:

\code
// Find memoryTypeIndex for the pool.
VkBufferCreateInfo sampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
sampleBufCreateInfo.size = 0x10000; // Doesn't matter.
sampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo sampleAllocCreateInfo = {};
sampleAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

uint32_t memTypeIndex;
VkResult res = vmaFindMemoryTypeIndexForBufferInfo(allocator,
    &sampleBufCreateInfo, &sampleAllocCreateInfo, &memTypeIndex);
// Check res...

// Create a pool that can have at most 2 blocks, 128 MiB each.
VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
poolCreateInfo.blockSize = 128ull * 1024 * 1024;
poolCreateInfo.maxBlockCount = 2;

VmaPool pool;
res = vmaCreatePool(allocator, &poolCreateInfo, &pool);
// Check res...

// Allocate a buffer out of it.
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 1024;
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.pool = pool;

VkBuffer buf;
VmaAllocation alloc;
res = vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, nullptr);
// Check res...
\endcode

You have to free all allocations made from this pool before destroying it.

\code
vmaDestroyBuffer(allocator, buf, alloc);
vmaDestroyPool(allocator, pool);
\endcode

New versions of this library support creating dedicated allocations in custom pools.
It is supported only when VmaPoolCreateInfo::blockSize = 0.
To use this feature, set VmaAllocationCreateInfo::pool to the pointer to your custom pool and
VmaAllocationCreateInfo::flags to #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.

\note Excessive use of custom pools is a common mistake when using this library.
Custom pools may be useful for special purposes - when you want to
keep certain type of resources separate e.g. to reserve minimum amount of memory
for them or limit maximum amount of memory they can occupy. For most
resources this is not needed and so it is not recommended to create #VmaPool
objects and allocations out of them. Allocating from the default pool is sufficient.


\section custom_memory_pools_MemTypeIndex Choosing memory type index

When creating a pool, you must explicitly specify memory type index.
To find the one suitable for your buffers or images, you can use helper functions
vmaFindMemoryTypeIndexForBufferInfo(), vmaFindMemoryTypeIndexForImageInfo().
You need to provide structures with example parameters of buffers or images
that you are going to create in that pool.

\code
VkBufferCreateInfo exampleBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
exampleBufCreateInfo.size = 1024; // Doesn't matter
exampleBufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

uint32_t memTypeIndex;
vmaFindMemoryTypeIndexForBufferInfo(allocator, &exampleBufCreateInfo, &allocCreateInfo, &memTypeIndex);

VmaPoolCreateInfo poolCreateInfo = {};
poolCreateInfo.memoryTypeIndex = memTypeIndex;
// ...
\endcode

When creating buffers/images allocated in that pool, provide following parameters:

- `VkBufferCreateInfo`: Prefer to pass same parameters as above.
  Otherwise you risk creating resources in a memory type that is not suitable for them, which may result in undefined behavior.
  Using different `VK_BUFFER_USAGE_` flags may work, but you shouldn't create images in a pool intended for buffers
  or the other way around.
- VmaAllocationCreateInfo: You don't need to pass same parameters. Fill only `pool` member.
  Other members are ignored anyway.

\section linear_algorithm Linear allocation algorithm

Each Vulkan memory block managed by this library has accompanying metadata that
keeps track of used and unused regions. By default, the metadata structure and
algorithm tries to find best place for new allocations among free regions to
optimize memory usage. This way you can allocate and free objects in any order.

![Default allocation algorithm](../gfx/Linear_allocator_1_algo_default.png)

Sometimes there is a need to use simpler, linear allocation algorithm. You can
create custom pool that uses such algorithm by adding flag
#VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT to VmaPoolCreateInfo::flags while creating
#VmaPool object. Then an alternative metadata management is used. It always
creates new allocations after last one and doesn't reuse free regions after
allocations freed in the middle. It results in better allocation performance and
less memory consumed by metadata.

![Linear allocation algorithm](../gfx/Linear_allocator_2_algo_linear.png)

With this one flag, you can create a custom pool that can be used in many ways:
free-at-once, stack, double stack, and ring buffer. See below for details.
You don't need to specify explicitly which of these options you are going to use - it is detected automatically.

\subsection linear_algorithm_free_at_once Free-at-once

In a pool that uses linear algorithm, you still need to free all the allocations
individually, e.g. by using vmaFreeMemory() or vmaDestroyBuffer(). You can free
them in any order. New allocations are always made after last one - free space
in the middle is not reused. However, when you release all the allocation and
the pool becomes empty, allocation starts from the beginning again. This way you
can use linear algorithm to speed up creation of allocations that you are going
to release all at once.

![Free-at-once](../gfx/Linear_allocator_3_free_at_once.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_stack Stack

When you free an allocation that was created last, its space can be reused.
Thanks to this, if you always release allocations in the order opposite to their
creation (LIFO - Last In First Out), you can achieve behavior of a stack.

![Stack](../gfx/Linear_allocator_4_stack.png)

This mode is also available for pools created with VmaPoolCreateInfo::maxBlockCount
value that allows multiple memory blocks.

\subsection linear_algorithm_double_stack Double stack

The space reserved by a custom pool with linear algorithm may be used by two
stacks:

- First, default one, growing up from offset 0.
- Second, "upper" one, growing down from the end towards lower offsets.

To make allocation from the upper stack, add flag #VMA_ALLOCATION_CREATE_UPPER_ADDRESS_BIT
to VmaAllocationCreateInfo::flags.

![Double stack](../gfx/Linear_allocator_7_double_stack.png)

Double stack is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

When the two stacks' ends meet so there is not enough space between them for a
new allocation, such allocation fails with usual
`VK_ERROR_OUT_OF_DEVICE_MEMORY` error.

\subsection linear_algorithm_ring_buffer Ring buffer

When you free some allocations from the beginning and there is not enough free space
for a new one at the end of a pool, allocator's "cursor" wraps around to the
beginning and starts allocation there. Thanks to this, if you always release
allocations in the same order as you created them (FIFO - First In First Out),
you can achieve behavior of a ring buffer / queue.

![Ring buffer](../gfx/Linear_allocator_5_ring_buffer.png)

Ring buffer is available only in pools with one memory block -
VmaPoolCreateInfo::maxBlockCount must be 1. Otherwise behavior is undefined.

\note \ref defragmentation is not supported in custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT.


\page defragmentation Defragmentation

Interleaved allocations and deallocations of many objects of varying size can
cause fragmentation over time, which can lead to a situation where the library is unable
to find a continuous range of free memory for a new allocation despite there is
enough free space, just scattered across many small free ranges between existing
allocations.

To mitigate this problem, you can use defragmentation feature.
It doesn't happen automatically though and needs your cooperation,
because VMA is a low level library that only allocates memory.
It cannot recreate buffers and images in a new place as it doesn't remember the contents of `VkBufferCreateInfo` / `VkImageCreateInfo` structures.
It cannot copy their contents as it doesn't record any commands to a command buffer.

Example:

\code
VmaDefragmentationInfo defragInfo = {};
defragInfo.pool = myPool;
defragInfo.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;

VmaDefragmentationContext defragCtx;
VkResult res = vmaBeginDefragmentation(allocator, &defragInfo, &defragCtx);
// Check res...

for(;;)
{
    VmaDefragmentationPassMoveInfo pass;
    res = vmaBeginDefragmentationPass(allocator, defragCtx, &pass);
    if(res == VK_SUCCESS)
        break;
    else if(res != VK_INCOMPLETE)
        // Handle error...

    for(uint32_t i = 0; i < pass.moveCount; ++i)
    {
        // Inspect pass.pMoves[i].srcAllocation, identify what buffer/image it represents.
        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(allocator, pass.pMoves[i].srcAllocation, &allocInfo);
        MyEngineResourceData* resData = (MyEngineResourceData*)allocInfo.pUserData;

        // Recreate and bind this buffer/image at: pass.pMoves[i].dstMemory, pass.pMoves[i].dstOffset.
        VkImageCreateInfo imgCreateInfo = ...
        VkImage newImg;
        res = vkCreateImage(device, &imgCreateInfo, nullptr, &newImg);
        // Check res...
        res = vmaBindImageMemory(allocator, pass.pMoves[i].dstTmpAllocation, newImg);
        // Check res...

        // Issue a vkCmdCopyBuffer/vkCmdCopyImage to copy its content to the new place.
        vkCmdCopyImage(cmdBuf, resData->img, ..., newImg, ...);
    }

    // Make sure the copy commands finished executing.
    vkWaitForFences(...);

    // Destroy old buffers/images bound with pass.pMoves[i].srcAllocation.
    for(uint32_t i = 0; i < pass.moveCount; ++i)
    {
        // ...
        vkDestroyImage(device, resData->img, nullptr);
    }

    // Update appropriate descriptors to point to the new places...

    res = vmaEndDefragmentationPass(allocator, defragCtx, &pass);
    if(res == VK_SUCCESS)
        break;
    else if(res != VK_INCOMPLETE)
        // Handle error...
}

vmaEndDefragmentation(allocator, defragCtx, nullptr);
\endcode

Although functions like vmaCreateBuffer(), vmaCreateImage(), vmaDestroyBuffer(), vmaDestroyImage()
create/destroy an allocation and a buffer/image at once, these are just a shortcut for
creating the resource, allocating memory, and binding them together.
Defragmentation works on memory allocations only. You must handle the rest manually.
Defragmentation is an iterative process that should repreat "passes" as long as related functions
return `VK_INCOMPLETE` not `VK_SUCCESS`.
In each pass:

1. vmaBeginDefragmentationPass() function call:
   - Calculates and returns the list of allocations to be moved in this pass.
     Note this can be a time-consuming process.
   - Reserves destination memory for them by creating temporary destination allocations
     that you can query for their `VkDeviceMemory` + offset using vmaGetAllocationInfo().
2. Inside the pass, **you should**:
   - Inspect the returned list of allocations to be moved.
   - Create new buffers/images and bind them at the returned destination temporary allocations.
   - Copy data from source to destination resources if necessary.
   - Destroy the source buffers/images, but NOT their allocations.
3. vmaEndDefragmentationPass() function call:
   - Frees the source memory reserved for the allocations that are moved.
   - Modifies source #VmaAllocation objects that are moved to point to the destination reserved memory.
   - Frees `VkDeviceMemory` blocks that became empty.

Unlike in previous iterations of the defragmentation API, there is no list of "movable" allocations passed as a parameter.
Defragmentation algorithm tries to move all suitable allocations.
You can, however, refuse to move some of them inside a defragmentation pass, by setting
`pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
This is not recommended and may result in suboptimal packing of the allocations after defragmentation.
If you cannot ensure any allocation can be moved, it is better to keep movable allocations separate in a custom pool.

Inside a pass, for each allocation that should be moved:

- You should copy its data from the source to the destination place by calling e.g. `vkCmdCopyBuffer()`, `vkCmdCopyImage()`.
  - You need to make sure these commands finished executing before destroying the source buffers/images and before calling vmaEndDefragmentationPass().
- If a resource doesn't contain any meaningful data, e.g. it is a transient color attachment image to be cleared,
  filled, and used temporarily in each rendering frame, you can just recreate this image
  without copying its data.
- If the resource is in `HOST_VISIBLE` and `HOST_CACHED` memory, you can copy its data on the CPU
  using `memcpy()`.
- If you cannot move the allocation, you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_IGNORE.
  This will cancel the move.
  - vmaEndDefragmentationPass() will then free the destination memory
    not the source memory of the allocation, leaving it unchanged.
- If you decide the allocation is unimportant and can be destroyed instead of moved (e.g. it wasn't used for long time),
  you can set `pass.pMoves[i].operation` to #VMA_DEFRAGMENTATION_MOVE_OPERATION_DESTROY.
  - vmaEndDefragmentationPass() will then free both source and destination memory, and will destroy the source #VmaAllocation object.

You can defragment a specific custom pool by setting VmaDefragmentationInfo::pool
(like in the example above) or all the default pools by setting this member to null.

Defragmentation is always performed in each pool separately.
Allocations are never moved between different Vulkan memory types.
The size of the destination memory reserved for a moved allocation is the same as the original one.
Alignment of an allocation as it was determined using `vkGetBufferMemoryRequirements()` etc. is also respected after defragmentation.
Buffers/images should be recreated with the same `VkBufferCreateInfo` / `VkImageCreateInfo` parameters as the original ones.

You can perform the defragmentation incrementally to limit the number of allocations and bytes to be moved
in each pass, e.g. to call it in sync with render frames and not to experience too big hitches.
See members: VmaDefragmentationInfo::maxBytesPerPass, VmaDefragmentationInfo::maxAllocationsPerPass.

It is also safe to perform the defragmentation asynchronously to render frames and other Vulkan and VMA
usage, possibly from multiple threads, with the exception that allocations
returned in VmaDefragmentationPassMoveInfo::pMoves shouldn't be destroyed until the defragmentation pass is ended.

<b>Mapping</b> is preserved on allocations that are moved during defragmentation.
Whether through #VMA_ALLOCATION_CREATE_MAPPED_BIT or vmaMapMemory(), the allocations
are mapped at their new place. Of course, pointer to the mapped data changes, so it needs to be queried
using VmaAllocationInfo::pMappedData.

\note Defragmentation is not supported in custom pools created with #VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT.


\page statistics Statistics

This library contains several functions that return information about its internal state,
especially the amount of memory allocated from Vulkan.

\section statistics_numeric_statistics Numeric statistics

If you need to obtain basic statistics about memory usage per heap, together with current budget,
you can call function vmaGetHeapBudgets() and inspect structure #VmaBudget.
This is useful to keep track of memory usage and stay within budget
(see also \ref staying_within_budget).
Example:

\code
uint32_t heapIndex = ...

VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
vmaGetHeapBudgets(allocator, budgets);

printf("My heap currently has %u allocations taking %llu B,\n",
    budgets[heapIndex].statistics.allocationCount,
    budgets[heapIndex].statistics.allocationBytes);
printf("allocated out of %u Vulkan device memory blocks taking %llu B,\n",
    budgets[heapIndex].statistics.blockCount,
    budgets[heapIndex].statistics.blockBytes);
printf("Vulkan reports total usage %llu B with budget %llu B.\n",
    budgets[heapIndex].usage,
    budgets[heapIndex].budget);
\endcode

You can query for more detailed statistics per memory heap, type, and totals,
including minimum and maximum allocation size and unused range size,
by calling function vmaCalculateStatistics() and inspecting structure #VmaTotalStatistics.
This function is slower though, as it has to traverse all the internal data structures,
so it should be used only for debugging purposes.

You can query for statistics of a custom pool using function vmaGetPoolStatistics()
or vmaCalculatePoolStatistics().

You can query for information about a specific allocation using function vmaGetAllocationInfo().
It fill structure #VmaAllocationInfo.

\section statistics_json_dump JSON dump

You can dump internal state of the allocator to a string in JSON format using function vmaBuildStatsString().
The result is guaranteed to be correct JSON.
It uses ANSI encoding.
Any strings provided by user (see [Allocation names](@ref allocation_names))
are copied as-is and properly escaped for JSON, so if they use UTF-8, ISO-8859-2 or any other encoding,
this JSON string can be treated as using this encoding.
It must be freed using function vmaFreeStatsString().

The format of this JSON string is not part of official documentation of the library,
but it will not change in backward-incompatible way without increasing library major version number
and appropriate mention in changelog.

The JSON string contains all the data that can be obtained using vmaCalculateStatistics().
It can also contain detailed map of allocated memory blocks and their regions -
free and occupied by allocations.
This allows e.g. to visualize the memory or assess fragmentation.


\page allocation_annotation Allocation names and user data

\section allocation_user_data Allocation user data

You can annotate allocations with your own information, e.g. for debugging purposes.
To do that, fill VmaAllocationCreateInfo::pUserData field when creating
an allocation. It is an opaque `void*` pointer. You can use it e.g. as a pointer,
some handle, index, key, ordinal number or any other value that would associate
the allocation with your custom metadata.
It is useful to identify appropriate data structures in your engine given #VmaAllocation,
e.g. when doing \ref defragmentation.

\code
VkBufferCreateInfo bufCreateInfo = ...

MyBufferMetadata* pMetadata = CreateBufferMetadata();

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.pUserData = pMetadata;

VkBuffer buffer;
VmaAllocation allocation;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buffer, &allocation, nullptr);
\endcode

The pointer may be later retrieved as VmaAllocationInfo::pUserData:

\code
VmaAllocationInfo allocInfo;
vmaGetAllocationInfo(allocator, allocation, &allocInfo);
MyBufferMetadata* pMetadata = (MyBufferMetadata*)allocInfo.pUserData;
\endcode

It can also be changed using function vmaSetAllocationUserData().

Values of (non-zero) allocations' `pUserData` are printed in JSON report created by
vmaBuildStatsString() in hexadecimal form.

\section allocation_names Allocation names

An allocation can also carry a null-terminated string, giving a name to the allocation.
To set it, call vmaSetAllocationName().
The library creates internal copy of the string, so the pointer you pass doesn't need
to be valid for whole lifetime of the allocation. You can free it after the call.

\code
std::string imageName = "Texture: ";
imageName += fileName;
vmaSetAllocationName(allocator, allocation, imageName.c_str());
\endcode

The string can be later retrieved by inspecting VmaAllocationInfo::pName.
It is also printed in JSON report created by vmaBuildStatsString().

\note Setting string name to VMA allocation doesn't automatically set it to the Vulkan buffer or image created with it.
You must do it manually using an extension like VK_EXT_debug_utils, which is independent of this library.


\page virtual_allocator Virtual allocator

As an extra feature, the core allocation algorithm of the library is exposed through a simple and convenient API of "virtual allocator".
It doesn't allocate any real GPU memory. It just keeps track of used and free regions of a "virtual block".
You can use it to allocate your own memory or other objects, even completely unrelated to Vulkan.
A common use case is sub-allocation of pieces of one large GPU buffer.

\section virtual_allocator_creating_virtual_block Creating virtual block

To use this functionality, there is no main "allocator" object.
You don't need to have #VmaAllocator object created.
All you need to do is to create a separate #VmaVirtualBlock object for each block of memory you want to be managed by the allocator:

-# Fill in #VmaVirtualBlockCreateInfo structure.
-# Call vmaCreateVirtualBlock(). Get new #VmaVirtualBlock object.

Example:

\code
VmaVirtualBlockCreateInfo blockCreateInfo = {};
blockCreateInfo.size = 1048576; // 1 MB

VmaVirtualBlock block;
VkResult res = vmaCreateVirtualBlock(&blockCreateInfo, &block);
\endcode

\section virtual_allocator_making_virtual_allocations Making virtual allocations

#VmaVirtualBlock object contains internal data structure that keeps track of free and occupied regions
using the same code as the main Vulkan memory allocator.
Similarly to #VmaAllocation for standard GPU allocations, there is #VmaVirtualAllocation type
that represents an opaque handle to an allocation within the virtual block.

In order to make such allocation:

-# Fill in #VmaVirtualAllocationCreateInfo structure.
-# Call vmaVirtualAllocate(). Get new #VmaVirtualAllocation object that represents the allocation.
   You can also receive `VkDeviceSize offset` that was assigned to the allocation.

Example:

\code
VmaVirtualAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.size = 4096; // 4 KB

VmaVirtualAllocation alloc;
VkDeviceSize offset;
res = vmaVirtualAllocate(block, &allocCreateInfo, &alloc, &offset);
if(res == VK_SUCCESS)
{
    // Use the 4 KB of your memory starting at offset.
}
else
{
    // Allocation failed - no space for it could be found. Handle this error!
}
\endcode

\section virtual_allocator_deallocation Deallocation

When no longer needed, an allocation can be freed by calling vmaVirtualFree().
You can only pass to this function an allocation that was previously returned by vmaVirtualAllocate()
called for the same #VmaVirtualBlock.

When whole block is no longer needed, the block object can be released by calling vmaDestroyVirtualBlock().
All allocations must be freed before the block is destroyed, which is checked internally by an assert.
However, if you don't want to call vmaVirtualFree() for each allocation, you can use vmaClearVirtualBlock() to free them all at once -
a feature not available in normal Vulkan memory allocator. Example:

\code
vmaVirtualFree(block, alloc);
vmaDestroyVirtualBlock(block);
\endcode

\section virtual_allocator_allocation_parameters Allocation parameters

You can attach a custom pointer to each allocation by using vmaSetVirtualAllocationUserData().
Its default value is null.
It can be used to store any data that needs to be associated with that allocation - e.g. an index, a handle, or a pointer to some
larger data structure containing more information. Example:

\code
struct CustomAllocData
{
    std::string m_AllocName;
};
CustomAllocData* allocData = new CustomAllocData();
allocData->m_AllocName = "My allocation 1";
vmaSetVirtualAllocationUserData(block, alloc, allocData);
\endcode

The pointer can later be fetched, along with allocation offset and size, by passing the allocation handle to function
vmaGetVirtualAllocationInfo() and inspecting returned structure #VmaVirtualAllocationInfo.
If you allocated a new object to be used as the custom pointer, don't forget to delete that object before freeing the allocation!
Example:

\code
VmaVirtualAllocationInfo allocInfo;
vmaGetVirtualAllocationInfo(block, alloc, &allocInfo);
delete (CustomAllocData*)allocInfo.pUserData;

vmaVirtualFree(block, alloc);
\endcode

\section virtual_allocator_alignment_and_units Alignment and units

It feels natural to express sizes and offsets in bytes.
If an offset of an allocation needs to be aligned to a multiply of some number (e.g. 4 bytes), you can fill optional member
VmaVirtualAllocationCreateInfo::alignment to request it. Example:

\code
VmaVirtualAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.size = 4096; // 4 KB
allocCreateInfo.alignment = 4; // Returned offset must be a multiply of 4 B

VmaVirtualAllocation alloc;
res = vmaVirtualAllocate(block, &allocCreateInfo, &alloc, nullptr);
\endcode

Alignments of different allocations made from one block may vary.
However, if all alignments and sizes are always multiply of some size e.g. 4 B or `sizeof(MyDataStruct)`,
you can express all sizes, alignments, and offsets in multiples of that size instead of individual bytes.
It might be more convenient, but you need to make sure to use this new unit consistently in all the places:

- VmaVirtualBlockCreateInfo::size
- VmaVirtualAllocationCreateInfo::size and VmaVirtualAllocationCreateInfo::alignment
- Using offset returned by vmaVirtualAllocate() or in VmaVirtualAllocationInfo::offset

\section virtual_allocator_statistics Statistics

You can obtain statistics of a virtual block using vmaGetVirtualBlockStatistics()
(to get brief statistics that are fast to calculate)
or vmaCalculateVirtualBlockStatistics() (to get more detailed statistics, slower to calculate).
The functions fill structures #VmaStatistics, #VmaDetailedStatistics respectively - same as used by the normal Vulkan memory allocator.
Example:

\code
VmaStatistics stats;
vmaGetVirtualBlockStatistics(block, &stats);
printf("My virtual block has %llu bytes used by %u virtual allocations\n",
    stats.allocationBytes, stats.allocationCount);
\endcode

You can also request a full list of allocations and free regions as a string in JSON format by calling
vmaBuildVirtualBlockStatsString().
Returned string must be later freed using vmaFreeVirtualBlockStatsString().
The format of this string differs from the one returned by the main Vulkan allocator, but it is similar.

\section virtual_allocator_additional_considerations Additional considerations

The "virtual allocator" functionality is implemented on a level of individual memory blocks.
Keeping track of a whole collection of blocks, allocating new ones when out of free space,
deleting empty ones, and deciding which one to try first for a new allocation must be implemented by the user.

Alternative allocation algorithms are supported, just like in custom pools of the real GPU memory.
See enum #VmaVirtualBlockCreateFlagBits to learn how to specify them (e.g. #VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT).
You can find their description in chapter \ref custom_memory_pools.
Allocation strategies are also supported.
See enum #VmaVirtualAllocationCreateFlagBits to learn how to specify them (e.g. #VMA_VIRTUAL_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT).

Following features are supported only by the allocator of the real GPU memory and not by virtual allocations:
buffer-image granularity, `VMA_DEBUG_MARGIN`, `VMA_MIN_ALIGNMENT`.


\page debugging_memory_usage Debugging incorrect memory usage

If you suspect a bug with memory usage, like usage of uninitialized memory or
memory being overwritten out of bounds of an allocation,
you can use debug features of this library to verify this.

\section debugging_memory_usage_initialization Memory initialization

If you experience a bug with incorrect and nondeterministic data in your program and you suspect uninitialized memory to be used,
you can enable automatic memory initialization to verify this.
To do it, define macro `VMA_DEBUG_INITIALIZE_ALLOCATIONS` to 1.

\code
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#include "vk_mem_alloc.h"
\endcode

It makes memory of new allocations initialized to bit pattern `0xDCDCDCDC`.
Before an allocation is destroyed, its memory is filled with bit pattern `0xEFEFEFEF`.
Memory is automatically mapped and unmapped if necessary.

If you find these values while debugging your program, good chances are that you incorrectly
read Vulkan memory that is allocated but not initialized, or already freed, respectively.

Memory initialization works only with memory types that are `HOST_VISIBLE` and with allocations that can be mapped.
It works also with dedicated allocations.

\section debugging_memory_usage_margins Margins

By default, allocations are laid out in memory blocks next to each other if possible
(considering required alignment, `bufferImageGranularity`, and `nonCoherentAtomSize`).

![Allocations without margin](../gfx/Margins_1.png)

Define macro `VMA_DEBUG_MARGIN` to some non-zero value (e.g. 16) to enforce specified
number of bytes as a margin after every allocation.

\code
#define VMA_DEBUG_MARGIN 16
#include "vk_mem_alloc.h"
\endcode

![Allocations with margin](../gfx/Margins_2.png)

If your bug goes away after enabling margins, it means it may be caused by memory
being overwritten outside of allocation boundaries. It is not 100% certain though.
Change in application behavior may also be caused by different order and distribution
of allocations across memory blocks after margins are applied.

Margins work with all types of memory.

Margin is applied only to allocations made out of memory blocks and not to dedicated
allocations, which have their own memory block of specific size.
It is thus not applied to allocations made using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT flag
or those automatically decided to put into dedicated allocations, e.g. due to its
large size or recommended by VK_KHR_dedicated_allocation extension.

Margins appear in [JSON dump](@ref statistics_json_dump) as part of free space.

Note that enabling margins increases memory usage and fragmentation.

Margins do not apply to \ref virtual_allocator.

\section debugging_memory_usage_corruption_detection Corruption detection

You can additionally define macro `VMA_DEBUG_DETECT_CORRUPTION` to 1 to enable validation
of contents of the margins.

\code
#define VMA_DEBUG_MARGIN 16
#define VMA_DEBUG_DETECT_CORRUPTION 1
#include "vk_mem_alloc.h"
\endcode

When this feature is enabled, number of bytes specified as `VMA_DEBUG_MARGIN`
(it must be multiply of 4) after every allocation is filled with a magic number.
This idea is also know as "canary".
Memory is automatically mapped and unmapped if necessary.

This number is validated automatically when the allocation is destroyed.
If it is not equal to the expected value, `VMA_ASSERT()` is executed.
It clearly means that either CPU or GPU overwritten the memory outside of boundaries of the allocation,
which indicates a serious bug.

You can also explicitly request checking margins of all allocations in all memory blocks
that belong to specified memory types by using function vmaCheckCorruption(),
or in memory blocks that belong to specified custom pool, by using function
vmaCheckPoolCorruption().

Margin validation (corruption detection) works only for memory types that are
`HOST_VISIBLE` and `HOST_COHERENT`.


\page opengl_interop OpenGL Interop

VMA provides some features that help with interoperability with OpenGL.

\section opengl_interop_exporting_memory Exporting memory

If you want to attach `VkExportMemoryAllocateInfoKHR` structure to `pNext` chain of memory allocations made by the library:

It is recommended to create \ref custom_memory_pools for such allocations.
Define and fill in your `VkExportMemoryAllocateInfoKHR` structure and attach it to VmaPoolCreateInfo::pMemoryAllocateNext
while creating the custom pool.
Please note that the structure must remain alive and unchanged for the whole lifetime of the #VmaPool,
not only while creating it, as no copy of the structure is made,
but its original pointer is used for each allocation instead.

If you want to export all memory allocated by the library from certain memory types,
also dedicated allocations or other allocations made from default pools,
an alternative solution is to fill in VmaAllocatorCreateInfo::pTypeExternalMemoryHandleTypes.
It should point to an array with `VkExternalMemoryHandleTypeFlagsKHR` to be automatically passed by the library
through `VkExportMemoryAllocateInfoKHR` on each allocation made from a specific memory type.
Please note that new versions of the library also support dedicated allocations created in custom pools.

You should not mix these two methods in a way that allows to apply both to the same memory type.
Otherwise, `VkExportMemoryAllocateInfoKHR` structure would be attached twice to the `pNext` chain of `VkMemoryAllocateInfo`.


\section opengl_interop_custom_alignment Custom alignment

Buffers or images exported to a different API like OpenGL may require a different alignment,
higher than the one used by the library automatically, queried from functions like `vkGetBufferMemoryRequirements`.
To impose such alignment:

It is recommended to create \ref custom_memory_pools for such allocations.
Set VmaPoolCreateInfo::minAllocationAlignment member to the minimum alignment required for each allocation
to be made out of this pool.
The alignment actually used will be the maximum of this member and the alignment returned for the specific buffer or image
from a function like `vkGetBufferMemoryRequirements`, which is called by VMA automatically.

If you want to create a buffer with a specific minimum alignment out of default pools,
use special function vmaCreateBufferWithAlignment(), which takes additional parameter `minAlignment`.

Note the problem of alignment affects only resources placed inside bigger `VkDeviceMemory` blocks and not dedicated
allocations, as these, by definition, always have alignment = 0 because the resource is bound to the beginning of its dedicated block.
Contrary to Direct3D 12, Vulkan doesn't have a concept of alignment of the entire memory block passed on its allocation.


\page usage_patterns Recommended usage patterns

Vulkan gives great flexibility in memory allocation.
This chapter shows the most common patterns.

See also slides from talk:
[Sawicki, Adam. Advanced Graphics Techniques Tutorial: Memory management in Vulkan and DX12. Game Developers Conference, 2018](https://www.gdcvault.com/play/1025458/Advanced-Graphics-Techniques-Tutorial-New)


\section usage_patterns_gpu_only GPU-only resource

<b>When:</b>
Any resources that you frequently write and read on GPU,
e.g. images used as color attachments (aka "render targets"), depth-stencil attachments,
images/buffers used as storage image/buffer (aka "Unordered Access View (UAV)").

<b>What to do:</b>
Let the library select the optimal memory type, which will likely have `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

\code
VkImageCreateInfo imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
imgCreateInfo.extent.width = 3840;
imgCreateInfo.extent.height = 2160;
imgCreateInfo.extent.depth = 1;
imgCreateInfo.mipLevels = 1;
imgCreateInfo.arrayLayers = 1;
imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imgCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
allocCreateInfo.priority = 1.0f;

VkImage img;
VmaAllocation alloc;
vmaCreateImage(allocator, &imgCreateInfo, &allocCreateInfo, &img, &alloc, nullptr);
\endcode

<b>Also consider:</b>
Consider creating them as dedicated allocations using #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
especially if they are large or if you plan to destroy and recreate them with different sizes
e.g. when display resolution changes.
Prefer to create such resources first and all other GPU resources (like textures and vertex buffers) later.
When VK_EXT_memory_priority extension is enabled, it is also worth setting high priority to such allocation
to decrease chances to be evicted to system memory by the operating system.

\section usage_patterns_staging_copy_upload Staging copy for upload

<b>When:</b>
A "staging" buffer than you want to map and fill from CPU code, then use as a source of transfer
to some GPU resource.

<b>What to do:</b>
Use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT.
Let the library select the optimal memory type, which will always have `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

...

memcpy(allocInfo.pMappedData, myData, myDataSize);
\endcode

<b>Also consider:</b>
You can map the allocation using vmaMapMemory() or you can create it as persistenly mapped
using #VMA_ALLOCATION_CREATE_MAPPED_BIT, as in the example above.


\section usage_patterns_readback Readback

<b>When:</b>
Buffers for data written by or transferred from the GPU that you want to read back on the CPU,
e.g. results of some computations.

<b>What to do:</b>
Use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT.
Let the library select the optimal memory type, which will always have `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`
and `VK_MEMORY_PROPERTY_HOST_CACHED_BIT`.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

...

const float* downloadedData = (const float*)allocInfo.pMappedData;
\endcode


\section usage_patterns_advanced_data_uploading Advanced data uploading

For resources that you frequently write on CPU via mapped pointer and
frequently read on GPU e.g. as a uniform buffer (also called "dynamic"), multiple options are possible:

-# Easiest solution is to have one copy of the resource in `HOST_VISIBLE` memory,
   even if it means system RAM (not `DEVICE_LOCAL`) on systems with a discrete graphics card,
   and make the device reach out to that resource directly.
   - Reads performed by the device will then go through PCI Express bus.
     The performance of this access may be limited, but it may be fine depending on the size
     of this resource (whether it is small enough to quickly end up in GPU cache) and the sparsity
     of access.
-# On systems with unified memory (e.g. AMD APU or Intel integrated graphics, mobile chips),
   a memory type may be available that is both `HOST_VISIBLE` (available for mapping) and `DEVICE_LOCAL`
   (fast to access from the GPU). Then, it is likely the best choice for such type of resource.
-# Systems with a discrete graphics card and separate video memory may or may not expose
   a memory type that is both `HOST_VISIBLE` and `DEVICE_LOCAL`, also known as Base Address Register (BAR).
   If they do, it represents a piece of VRAM (or entire VRAM, if ReBAR is enabled in the motherboard BIOS)
   that is available to CPU for mapping.
   - Writes performed by the host to that memory go through PCI Express bus.
     The performance of these writes may be limited, but it may be fine, especially on PCIe 4.0,
     as long as rules of using uncached and write-combined memory are followed - only sequential writes and no reads.
-# Finally, you may need or prefer to create a separate copy of the resource in `DEVICE_LOCAL` memory,
   a separate "staging" copy in `HOST_VISIBLE` memory and perform an explicit transfer command between them.

Thankfully, VMA offers an aid to create and use such resources in the the way optimal
for the current Vulkan device. To help the library make the best choice,
use flag #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT together with
#VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT.
It will then prefer a memory type that is both `DEVICE_LOCAL` and `HOST_VISIBLE` (integrated memory or BAR),
but if no such memory type is available or allocation from it fails
(PC graphics cards have only 256 MB of BAR by default, unless ReBAR is supported and enabled in BIOS),
it will fall back to `DEVICE_LOCAL` memory for fast GPU access.
It is then up to you to detect that the allocation ended up in a memory type that is not `HOST_VISIBLE`,
so you need to create another "staging" allocation and perform explicit transfers.

\code
VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
bufCreateInfo.size = 65536;
bufCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;

VkBuffer buf;
VmaAllocation alloc;
VmaAllocationInfo allocInfo;
vmaCreateBuffer(allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

VkMemoryPropertyFlags memPropFlags;
vmaGetAllocationMemoryProperties(allocator, alloc, &memPropFlags);

if(memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
{
    // Allocation ended up in a mappable memory and is already mapped - write to it directly.

    // [Executed in runtime]:
    memcpy(allocInfo.pMappedData, myData, myDataSize);
}
else
{
    // Allocation ended up in a non-mappable memory - need to transfer.
    VkBufferCreateInfo stagingBufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingBufCreateInfo.size = 65536;
    stagingBufCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCreateInfo = {};
    stagingAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingAlloc;
    VmaAllocationInfo stagingAllocInfo;
    vmaCreateBuffer(allocator, &stagingBufCreateInfo, &stagingAllocCreateInfo,
        &stagingBuf, &stagingAlloc, stagingAllocInfo);

    // [Executed in runtime]:
    memcpy(stagingAllocInfo.pMappedData, myData, myDataSize);
    vmaFlushAllocation(allocator, stagingAlloc, 0, VK_WHOLE_SIZE);
    //vkCmdPipelineBarrier: VK_ACCESS_HOST_WRITE_BIT --> VK_ACCESS_TRANSFER_READ_BIT
    VkBufferCopy bufCopy = {
        0, // srcOffset
        0, // dstOffset,
        myDataSize); // size
    vkCmdCopyBuffer(cmdBuf, stagingBuf, buf, 1, &bufCopy);
}
\endcode

\section usage_patterns_other_use_cases Other use cases

Here are some other, less obvious use cases and their recommended settings:

- An image that is used only as transfer source and destination, but it should stay on the device,
  as it is used to temporarily store a copy of some texture, e.g. from the current to the next frame,
  for temporal antialiasing or other temporal effects.
  - Use `VkImageCreateInfo::usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO
- An image that is used only as transfer source and destination, but it should be placed
  in the system RAM despite it doesn't need to be mapped, because it serves as a "swap" copy to evict
  least recently used textures from VRAM.
  - Use `VkImageCreateInfo::usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
    as VMA needs a hint here to differentiate from the previous case.
- A buffer that you want to map and write from the CPU, directly read from the GPU
  (e.g. as a uniform or vertex buffer), but you have a clear preference to place it in device or
  host memory due to its large size.
  - Use `VkBufferCreateInfo::usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`
  - Use VmaAllocationCreateInfo::usage = #VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE or #VMA_MEMORY_USAGE_AUTO_PREFER_HOST
  - Use VmaAllocationCreateInfo::flags = #VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT


\page configuration Configuration

Please check "CONFIGURATION SECTION" in the code to find macros that you can define
before each include of this file or change directly in this file to provide
your own implementation of basic facilities like assert, `min()` and `max()` functions,
mutex, atomic etc.
The library uses its own implementation of containers by default, but you can switch to using
STL containers instead.

For example, define `VMA_ASSERT(expr)` before including the library to provide
custom implementation of the assertion, compatible with your project.
By default it is defined to standard C `assert(expr)` in `_DEBUG` configuration
and empty otherwise.

\section config_Vulkan_functions Pointers to Vulkan functions

There are multiple ways to import pointers to Vulkan functions in the library.
In the simplest case you don't need to do anything.
If the compilation or linking of your program or the initialization of the #VmaAllocator
doesn't work for you, you can try to reconfigure it.

First, the allocator tries to fetch pointers to Vulkan functions linked statically,
like this:

\code
m_VulkanFunctions.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
\endcode

If you want to disable this feature, set configuration macro: `#define VMA_STATIC_VULKAN_FUNCTIONS 0`.

Second, you can provide the pointers yourself by setting member VmaAllocatorCreateInfo::pVulkanFunctions.
You can fetch them e.g. using functions `vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` or
by using a helper library like [volk](https://github.com/zeux/volk).

Third, VMA tries to fetch remaining pointers that are still null by calling
`vkGetInstanceProcAddr` and `vkGetDeviceProcAddr` on its own.
You need to only fill in VmaVulkanFunctions::vkGetInstanceProcAddr and VmaVulkanFunctions::vkGetDeviceProcAddr.
Other pointers will be fetched automatically.
If you want to disable this feature, set configuration macro: `#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0`.

Finally, all the function pointers required by the library (considering selected
Vulkan version and enabled extensions) are checked with `VMA_ASSERT` if they are not null.


\section custom_memory_allocator Custom host memory allocator

If you use custom allocator for CPU memory rather than default operator `new`
and `delete` from C++, you can make this library using your allocator as well
by filling optional member VmaAllocatorCreateInfo::pAllocationCallbacks. These
functions will be passed to Vulkan, as well as used by the library itself to
make any CPU-side allocations.

\section allocation_callbacks Device memory allocation callbacks

The library makes calls to `vkAllocateMemory()` and `vkFreeMemory()` internally.
You can setup callbacks to be informed about these calls, e.g. for the purpose
of gathering some statistics. To do it, fill optional member
VmaAllocatorCreateInfo::pDeviceMemoryCallbacks.

\section heap_memory_limit Device heap memory limit

When device memory of certain heap runs out of free space, new allocations may
fail (returning error code) or they may succeed, silently pushing some existing_
memory blocks from GPU VRAM to system RAM (which degrades performance). This
behavior is implementation-dependent - it depends on GPU vendor and graphics
driver.

On AMD cards it can be controlled while creating Vulkan device object by using
VK_AMD_memory_overallocation_behavior extension, if available.

Alternatively, if you want to test how your program behaves with limited amount of Vulkan device
memory available without switching your graphics card to one that really has
smaller VRAM, you can use a feature of this library intended for this purpose.
To do it, fill optional member VmaAllocatorCreateInfo::pHeapSizeLimit.



\page vk_khr_dedicated_allocation VK_KHR_dedicated_allocation

VK_KHR_dedicated_allocation is a Vulkan extension which can be used to improve
performance on some GPUs. It augments Vulkan API with possibility to query
driver whether it prefers particular buffer or image to have its own, dedicated
allocation (separate `VkDeviceMemory` block) for better efficiency - to be able
to do some internal optimizations. The extension is supported by this library.
It will be used automatically when enabled.

It has been promoted to core Vulkan 1.1, so if you use eligible Vulkan version
and inform VMA about it by setting VmaAllocatorCreateInfo::vulkanApiVersion,
you are all set.

Otherwise, if you want to use it as an extension:

1 . When creating Vulkan device, check if following 2 device extensions are
supported (call `vkEnumerateDeviceExtensionProperties()`).
If yes, enable them (fill `VkDeviceCreateInfo::ppEnabledExtensionNames`).

- VK_KHR_get_memory_requirements2
- VK_KHR_dedicated_allocation

If you enabled these extensions:

2 . Use #VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT flag when creating
your #VmaAllocator to inform the library that you enabled required extensions
and you want the library to use them.

\code
allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;

vmaCreateAllocator(&allocatorInfo, &allocator);
\endcode

That is all. The extension will be automatically used whenever you create a
buffer using vmaCreateBuffer() or image using vmaCreateImage().

When using the extension together with Vulkan Validation Layer, you will receive
warnings like this:

_vkBindBufferMemory(): Binding memory to buffer 0x33 but vkGetBufferMemoryRequirements() has not been called on that buffer._

It is OK, you should just ignore it. It happens because you use function
`vkGetBufferMemoryRequirements2KHR()` instead of standard
`vkGetBufferMemoryRequirements()`, while the validation layer seems to be
unaware of it.

To learn more about this extension, see:

- [VK_KHR_dedicated_allocation in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap50.html#VK_KHR_dedicated_allocation)
- [VK_KHR_dedicated_allocation unofficial manual](http://asawicki.info/articles/VK_KHR_dedicated_allocation.php5)



\page vk_ext_memory_priority VK_EXT_memory_priority

VK_EXT_memory_priority is a device extension that allows to pass additional "priority"
value to Vulkan memory allocations that the implementation may use prefer certain
buffers and images that are critical for performance to stay in device-local memory
in cases when the memory is over-subscribed, while some others may be moved to the system memory.

VMA offers convenient usage of this extension.
If you enable it, you can pass "priority" parameter when creating allocations or custom pools
and the library automatically passes the value to Vulkan using this extension.

If you want to use this extension in connection with VMA, follow these steps:

\section vk_ext_memory_priority_initialization Initialization

1) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains "VK_EXT_memory_priority".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceMemoryPriorityFeaturesEXT` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceMemoryPriorityFeaturesEXT::memoryPriority` is true.

3) While creating device with `vkCreateDevice`, enable this extension - add "VK_EXT_memory_priority"
to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceMemoryPriorityFeaturesEXT` to
`VkPhysicalDeviceFeatures2::pNext` chain and set its member `memoryPriority` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this extension and feature - add #VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT
to VmaAllocatorCreateInfo::flags.

\section vk_ext_memory_priority_usage Usage

When using this extension, you should initialize following member:

- VmaAllocationCreateInfo::priority when creating a dedicated allocation with #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
- VmaPoolCreateInfo::priority when creating a custom pool.

It should be a floating-point value between `0.0f` and `1.0f`, where recommended default is `0.5f`.
Memory allocated with higher value can be treated by the Vulkan implementation as higher priority
and so it can have lower chances of being pushed out to system memory, experiencing degraded performance.

It might be a good idea to create performance-critical resources like color-attachment or depth-stencil images
as dedicated and set high priority to them. For example:

\code
VkImageCreateInfo imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
imgCreateInfo.extent.width = 3840;
imgCreateInfo.extent.height = 2160;
imgCreateInfo.extent.depth = 1;
imgCreateInfo.mipLevels = 1;
imgCreateInfo.arrayLayers = 1;
imgCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imgCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VmaAllocationCreateInfo allocCreateInfo = {};
allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
allocCreateInfo.priority = 1.0f;

VkImage img;
VmaAllocation alloc;
vmaCreateImage(allocator, &imgCreateInfo, &allocCreateInfo, &img, &alloc, nullptr);
\endcode

`priority` member is ignored in the following situations:

- Allocations created in custom pools: They inherit the priority, along with all other allocation parameters
  from the parametrs passed in #VmaPoolCreateInfo when the pool was created.
- Allocations created in default pools: They inherit the priority from the parameters
  VMA used when creating default pools, which means `priority == 0.5f`.


\page vk_amd_device_coherent_memory VK_AMD_device_coherent_memory

VK_AMD_device_coherent_memory is a device extension that enables access to
additional memory types with `VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD` and
`VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` flag. It is useful mostly for
allocation of buffers intended for writing "breadcrumb markers" in between passes
or draw calls, which in turn are useful for debugging GPU crash/hang/TDR cases.

When the extension is available but has not been enabled, Vulkan physical device
still exposes those memory types, but their usage is forbidden. VMA automatically
takes care of that - it returns `VK_ERROR_FEATURE_NOT_PRESENT` when an attempt
to allocate memory of such type is made.

If you want to use this extension in connection with VMA, follow these steps:

\section vk_amd_device_coherent_memory_initialization Initialization

1) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains "VK_AMD_device_coherent_memory".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceCoherentMemoryFeaturesAMD` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceCoherentMemoryFeaturesAMD::deviceCoherentMemory` is true.

3) While creating device with `vkCreateDevice`, enable this extension - add "VK_AMD_device_coherent_memory"
to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceCoherentMemoryFeaturesAMD` to
`VkPhysicalDeviceFeatures2::pNext` and set its member `deviceCoherentMemory` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this extension and feature - add #VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT
to VmaAllocatorCreateInfo::flags.

\section vk_amd_device_coherent_memory_usage Usage

After following steps described above, you can create VMA allocations and custom pools
out of the special `DEVICE_COHERENT` and `DEVICE_UNCACHED` memory types on eligible
devices. There are multiple ways to do it, for example:

- You can request or prefer to allocate out of such memory types by adding
  `VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD` to VmaAllocationCreateInfo::requiredFlags
  or VmaAllocationCreateInfo::preferredFlags. Those flags can be freely mixed with
  other ways of \ref choosing_memory_type, like setting VmaAllocationCreateInfo::usage.
- If you manually found memory type index to use for this purpose, force allocation
  from this specific index by setting VmaAllocationCreateInfo::memoryTypeBits `= 1u << index`.

\section vk_amd_device_coherent_memory_more_information More information

To learn more about this extension, see [VK_AMD_device_coherent_memory in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_device_coherent_memory.html)

Example use of this extension can be found in the code of the sample and test suite
accompanying this library.


\page enabling_buffer_device_address Enabling buffer device address

Device extension VK_KHR_buffer_device_address
allow to fetch raw GPU pointer to a buffer and pass it for usage in a shader code.
It has been promoted to core Vulkan 1.2.

If you want to use this feature in connection with VMA, follow these steps:

\section enabling_buffer_device_address_initialization Initialization

1) (For Vulkan version < 1.2) Call `vkEnumerateDeviceExtensionProperties` for the physical device.
Check if the extension is supported - if returned array of `VkExtensionProperties` contains
"VK_KHR_buffer_device_address".

2) Call `vkGetPhysicalDeviceFeatures2` for the physical device instead of old `vkGetPhysicalDeviceFeatures`.
Attach additional structure `VkPhysicalDeviceBufferDeviceAddressFeatures*` to `VkPhysicalDeviceFeatures2::pNext` to be returned.
Check if the device feature is really supported - check if `VkPhysicalDeviceBufferDeviceAddressFeatures::bufferDeviceAddress` is true.

3) (For Vulkan version < 1.2) While creating device with `vkCreateDevice`, enable this extension - add
"VK_KHR_buffer_device_address" to the list passed as `VkDeviceCreateInfo::ppEnabledExtensionNames`.

4) While creating the device, also don't set `VkDeviceCreateInfo::pEnabledFeatures`.
Fill in `VkPhysicalDeviceFeatures2` structure instead and pass it as `VkDeviceCreateInfo::pNext`.
Enable this device feature - attach additional structure `VkPhysicalDeviceBufferDeviceAddressFeatures*` to
`VkPhysicalDeviceFeatures2::pNext` and set its member `bufferDeviceAddress` to `VK_TRUE`.

5) While creating #VmaAllocator with vmaCreateAllocator() inform VMA that you
have enabled this feature - add #VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
to VmaAllocatorCreateInfo::flags.

\section enabling_buffer_device_address_usage Usage

After following steps described above, you can create buffers with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT*` using VMA.
The library automatically adds `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT*` to
allocated memory blocks wherever it might be needed.

Please note that the library supports only `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT*`.
The second part of this functionality related to "capture and replay" is not supported,
as it is intended for usage in debugging tools like RenderDoc, not in everyday Vulkan usage.

\section enabling_buffer_device_address_more_information More information

To learn more about this extension, see [VK_KHR_buffer_device_address in Vulkan specification](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap46.html#VK_KHR_buffer_device_address)

Example use of this extension can be found in the code of the sample and test suite
accompanying this library.

\page general_considerations General considerations

\section general_considerations_thread_safety Thread safety

- The library has no global state, so separate #VmaAllocator objects can be used
  independently.
  There should be no need to create multiple such objects though - one per `VkDevice` is enough.
- By default, all calls to functions that take #VmaAllocator as first parameter
  are safe to call from multiple threads simultaneously because they are
  synchronized internally when needed.
  This includes allocation and deallocation from default memory pool, as well as custom #VmaPool.
- When the allocator is created with #VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT
  flag, calls to functions that take such #VmaAllocator object must be
  synchronized externally.
- Access to a #VmaAllocation object must be externally synchronized. For example,
  you must not call vmaGetAllocationInfo() and vmaMapMemory() from different
  threads at the same time if you pass the same #VmaAllocation object to these
  functions.
- #VmaVirtualBlock is not safe to be used from multiple threads simultaneously.

\section general_considerations_versioning_and_compatibility Versioning and compatibility

The library uses [**Semantic Versioning**](https://semver.org/),
which means version numbers follow convention: Major.Minor.Patch (e.g. 2.3.0), where:

- Incremented Patch version means a release is backward- and forward-compatible,
  introducing only some internal improvements, bug fixes, optimizations etc.
  or changes that are out of scope of the official API described in this documentation.
- Incremented Minor version means a release is backward-compatible,
  so existing code that uses the library should continue to work, while some new
  symbols could have been added: new structures, functions, new values in existing
  enums and bit flags, new structure members, but not new function parameters.
- Incrementing Major version means a release could break some backward compatibility.

All changes between official releases are documented in file "CHANGELOG.md".

\warning Backward compatibility is considered on the level of C++ source code, not binary linkage.
Adding new members to existing structures is treated as backward compatible if initializing
the new members to binary zero results in the old behavior.
You should always fully initialize all library structures to zeros and not rely on their
exact binary size.

\section general_considerations_validation_layer_warnings Validation layer warnings

When using this library, you can meet following types of warnings issued by
Vulkan validation layer. They don't necessarily indicate a bug, so you may need
to just ignore them.

- *vkBindBufferMemory(): Binding memory to buffer 0xeb8e4 but vkGetBufferMemoryRequirements() has not been called on that buffer.*
  - It happens when VK_KHR_dedicated_allocation extension is enabled.
    `vkGetBufferMemoryRequirements2KHR` function is used instead, while validation layer seems to be unaware of it.
- *Mapping an image with layout VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL can result in undefined behavior if this memory is used by the device. Only GENERAL or PREINITIALIZED should be used.*
  - It happens when you map a buffer or image, because the library maps entire
    `VkDeviceMemory` block, where different types of images and buffers may end
    up together, especially on GPUs with unified memory like Intel.
- *Non-linear image 0xebc91 is aliased with linear buffer 0xeb8e4 which may indicate a bug.*
  - It may happen when you use [defragmentation](@ref defragmentation).

\section general_considerations_allocation_algorithm Allocation algorithm

The library uses following algorithm for allocation, in order:

-# Try to find free range of memory in existing blocks.
-# If failed, try to create a new block of `VkDeviceMemory`, with preferred block size.
-# If failed, try to create such block with size / 2, size / 4, size / 8.
-# If failed, try to allocate separate `VkDeviceMemory` for this allocation,
   just like when you use #VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT.
-# If failed, choose other memory type that meets the requirements specified in
   VmaAllocationCreateInfo and go to point 1.
-# If failed, return `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

\section general_considerations_features_not_supported Features not supported

Features deliberately excluded from the scope of this library:

-# **Data transfer.** Uploading (streaming) and downloading data of buffers and images
   between CPU and GPU memory and related synchronization is responsibility of the user.
   Defining some "texture" object that would automatically stream its data from a
   staging copy in CPU memory to GPU memory would rather be a feature of another,
   higher-level library implemented on top of VMA.
   VMA doesn't record any commands to a `VkCommandBuffer`. It just allocates memory.
-# **Recreation of buffers and images.** Although the library has functions for
   buffer and image creation: vmaCreateBuffer(), vmaCreateImage(), you need to
   recreate these objects yourself after defragmentation. That is because the big
   structures `VkBufferCreateInfo`, `VkImageCreateInfo` are not stored in
   #VmaAllocation object.
-# **Handling CPU memory allocation failures.** When dynamically creating small C++
   objects in CPU memory (not Vulkan memory), allocation failures are not checked
   and handled gracefully, because that would complicate code significantly and
   is usually not needed in desktop PC applications anyway.
   Success of an allocation is just checked with an assert.
-# **Code free of any compiler warnings.** Maintaining the library to compile and
   work correctly on so many different platforms is hard enough. Being free of
   any warnings, on any version of any compiler, is simply not feasible.
   There are many preprocessor macros that make some variables unused, function parameters unreferenced,
   or conditional expressions constant in some configurations.
   The code of this library should not be bigger or more complicated just to silence these warnings.
   It is recommended to disable such warnings instead.
-# This is a C++ library with C interface. **Bindings or ports to any other programming languages** are welcome as external projects but
   are not going to be included into this repository.
*/
