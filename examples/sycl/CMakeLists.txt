#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

add_compile_options(-I${PROJECT_SOURCE_DIR}/ggml)
add_compile_options(-I${PROJECT_SOURCE_DIR}/ggml/src)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

set(TARGET llama-ls-sycl-device)
add_executable(${TARGET} ls-sycl-device.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
