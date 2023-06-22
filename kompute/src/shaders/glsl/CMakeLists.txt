# SPDX-License-Identifier: Apache-2.0
# ######################
cmake_minimum_required(VERSION 3.20)

# Check if build shaders from source is enabled
if(KOMPUTE_OPT_BUILD_SHADERS)
    vulkan_compile_shader(INFILE ShaderOpMult.comp
        OUTFILE ShaderOpMult.hpp
        NAMESPACE "kp")

    vulkan_compile_shader(INFILE ShaderLogisticRegression.comp
        OUTFILE ShaderLogisticRegression.hpp
        NAMESPACE "kp")
else() # Else we will use our precompiled versions
    add_custom_command(OUTPUT $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderOpMult.hpp COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/ShaderOpMult.hpp.in $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderOpMult.hpp)
    add_custom_command(OUTPUT $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderLogisticRegression.hpp COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/ShaderLogisticRegression.hpp.in $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderLogisticRegression.hpp)
endif()

add_library(kp_shader INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/ShaderOpMult.hpp"
    "${CMAKE_CURRENT_BINARY_DIR}/ShaderLogisticRegression.hpp")

target_include_directories(kp_shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# Make sure we install shaders:
install(FILES $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderOpMult.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/ShaderLogisticRegression.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
