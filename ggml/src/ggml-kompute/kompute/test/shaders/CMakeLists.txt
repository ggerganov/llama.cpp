# SPDX-License-Identifier: Apache-2.0
# ######################
cmake_minimum_required(VERSION 3.20)

add_library(test_shaders "Utils.cpp"
    "Utils.hpp")

add_subdirectory(glsl)