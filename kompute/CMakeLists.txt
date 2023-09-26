# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(kompute VERSION 0.8.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# Only change the folder behavior if kompute is not a subproject
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
    set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
endif()

# Avoid the dll boilerplate code for windows
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

set(KOMPUTE_LIBRARIES kompute CACHE INTERNAL "")

# ####################################################
# Options
# ####################################################
macro(kompute_option OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    option(${OPTION_NAME} ${OPTION_TEXT} ${OPTION_DEFAULT})

    if(DEFINED ENV{${OPTION_NAME}})
        # Allow overriding the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()

    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()

    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

macro(kompute_log_level OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    set(${OPTION_NAME} ${OPTION_DEFAULT} CACHE STRING ${OPTION_TEXT})
    set_property(CACHE ${OPTION_NAME} PROPERTY STRINGS "Trace" "Debug" "Info" "Warn" "Error" "Critical" "Default" "Off")

    if(DEFINED ENV{${OPTION_NAME}})
        # Allow setting the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()

    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()

    # Allow disabling logging completely and prevent linking against it:
    if(${KOMPUTE_OPT_LOG_LEVEL} STREQUAL "Off")
        set(${OPTION_NAME}_DISABLED ON)
        add_compile_definitions(${OPTION_NAME}_DISABLED=1)
    endif()

    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

macro(kompute_option_string OPTION_NAME OPTION_TEXT OPTION_DEFAULT)
    set(${OPTION_NAME} ${OPTION_DEFAULT} CACHE STRING ${OPTION_TEXT})

    if(DEFINED ENV{${OPTION_NAME}})
        # Allow setting the option through an environment variable
        set(${OPTION_NAME} $ENV{${OPTION_NAME}})
    endif()

    if(${OPTION_NAME})
        add_definitions(-D${OPTION_NAME})
    endif()

    message(STATUS "  ${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

message(STATUS "General purpose GPU compute framework built on Vulkan")
message(STATUS "=======================================================")

# Build options
kompute_log_level(KOMPUTE_OPT_LOG_LEVEL "Internally we use Spdlog or fmt for logging, depending on the value of 'KOMPUTE_OPT_USE_SPDLOG'. The log level used can be changed here. Possible values: 'Trace', 'Debug', 'Info', 'Warn', 'Error', 'Critical', 'Off', 'Default'. If set to 'Off' logging will be deactivated completely. If set to 'Default', the log level will be set to 'Info' for release builds and 'Debug' else." "Off")
kompute_option(KOMPUTE_OPT_USE_SPDLOG "If enabled, logging via KP_LOG_<DEBUG, INFO, etc...> will happen through Spdlog instead of plan fmt." OFF)
kompute_option(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS "Explicitly disable debug layers even on debug." ON)
kompute_option(KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK "Whether to check if your driver supports the Vulkan Header version you are linking against. This might be useful in case you build shared on a different system than you run later." OFF)
kompute_option(KOMPUTE_OPT_BUILD_SHADERS "Rebuilds all compute shaders during compilation and does not use the already precompiled versions. Requires glslangValidator to be installed on your system." OFF)

# External components
kompute_option(KOMPUTE_OPT_USE_BUILT_IN_SPDLOG "Use the built-in version of Spdlog. Requires 'KOMPUTE_OPT_USE_SPDLOG' to be set to ON in order to have any effect." ON)
kompute_option(KOMPUTE_OPT_SPDLOG_ASYNC_MODE "If spdlog is enabled this allows for selecting whether the default logger setup creates sync or async logger" OFF)
kompute_option(KOMPUTE_OPT_USE_BUILT_IN_FMT "Use the built-in version of fmt." ON)
kompute_option(KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER "Use the built-in version of Vulkan Headers. This could be helpful in case your system Vulkan Headers are too new for your driver. If you set this to OFF, please make sure your system Vulkan Headers are supported by your driver." ON)
kompute_option_string(KOMPUTE_OPT_BUILT_IN_VULKAN_HEADER_TAG "The git tag used for the built-in Vulkan Headers when 'KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER' is enabled. A list of tags can be found here: https://github.com/KhronosGroup/Vulkan-Headers/tags" "v1.3.231")
message(STATUS "=======================================================")

# ####################################################
# Deprecated Options
# ####################################################
include(cmake/deprecation_warnings.cmake)

# ####################################################
# Dependencies
# ####################################################
include(cmake/vulkan_shader_compiler.cmake)
include(cmake/check_vulkan_version.cmake)
include(FetchContent)

# Vulkan Header
if(KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER)
    FetchContent_Declare(vulkan_header GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG ${KOMPUTE_OPT_BUILT_IN_VULKAN_HEADER_TAG}) # Source: https://github.com/KhronosGroup/Vulkan-Headers/tags
    FetchContent_MakeAvailable(vulkan_header)

    if(NOT KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK)
        # Ensure the driver supports this Vulkan version
        check_vulkan_version(INCLUDE_DIR "${vulkan_header_SOURCE_DIR}/include")
    endif()
endif()

find_package(Vulkan REQUIRED)

if(Vulkan_FOUND AND NOT TARGET Vulkan::Headers)
    add_library(Vulkan::Headers INTERFACE IMPORTED)
    set_target_properties(Vulkan::Headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIRS}")
endif()

if(NOT KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER AND NOT KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK)
    # Ensure the driver supports this Vulkan version
    check_vulkan_version(INCLUDE_DIR ${Vulkan_INCLUDE_DIR})
endif()

# Spdlog
if(KOMPUTE_OPT_USE_SPDLOG)
    add_compile_definitions(KOMPUTE_OPT_USE_SPDLOG=1)

    if(NOT KOMPUTE_OPT_LOG_LEVEL_DISABLED)
        if(KOMPUTE_OPT_USE_BUILT_IN_SPDLOG)
            set(SPDLOG_BUILD_SHARED ${BUILD_SHARED_LIBS})

            FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git
                GIT_TAG v1.10.0) # Source: https://github.com/gabime/spdlog/releases
            FetchContent_MakeAvailable(spdlog)
        else()
            find_package(spdlog REQUIRED)
        endif()
    endif()
endif()

# fmt
if(KOMPUTE_OPT_USE_BUILT_IN_FMT)
    FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.0.0) # Source: https://github.com/fmtlib/fmt/releases
    FetchContent_MakeAvailable(fmt)
else()
    find_package(fmt REQUIRED)
endif()

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

# ####################################################
# Preprocessor Macros
# ####################################################
if(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS)
    add_compile_definitions(KOMPUTE_DISABLE_VK_DEBUG_LAYERS=1)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -Wno-error=array-bounds")
endif()

# If glslang is cloned, then SPIRV/GlslangToSpv.h will be used instead of glslang/SPIRV/GlslangToSpv.h
# As after installation, SPIRV/ header files will be found in glslang/SPIRV/ , more info in #193
if(KOMPUTE_OPT_REPO_SUBMODULE_BUILD)
    add_definitions(-DUSE_EXTERNAL_GLSLANG)
endif()

# Allow scripts to call main kompute Makefile
function(kompute_make KOMPUTE_MAKE_TARGET)
    add_custom_target(${KOMPUTE_MAKE_TARGET}
        COMMAND make -C ${PROJECT_SOURCE_DIR} ${KOMPUTE_MAKE_TARGET})
endfunction()

add_executable(xxd external/bin/xxd.c)

add_subdirectory(src)
