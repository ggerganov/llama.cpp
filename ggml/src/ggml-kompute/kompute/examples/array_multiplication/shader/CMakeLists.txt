cmake_minimum_required(VERSION 3.20)

# To add more shaders simply copy the vulkan_compile_shader command and replace it with your new shader
vulkan_compile_shader(INFILE my_shader.comp
    OUTFILE my_shader.hpp
    NAMESPACE "shader"
    RELATIVE_PATH "${kompute_SOURCE_DIR}/cmake")

# Then add it to the library, so you can access it later in your code
add_library(shader INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/my_shader.hpp"

    # "${CMAKE_CURRENT_BINARY_DIR}/my_shader2.hpp"
)

target_include_directories(shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
