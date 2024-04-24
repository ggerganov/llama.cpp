cmake_minimum_required(VERSION 3.22.1)
project(ggml-qnn-test)

set(CMAKE_VERBOSE_MAKEFILE          on)
set(CMAKE_CXX_STANDARD              17)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set to ON if target Android phone is based on Qualcomm Snapdragon 8 Gen 3
set(TARGET_SNAPDRAGON_8_GEN3        OFF)

set(QNN_INC_PATH                    ${QNN_SDK_PATH}/include/QNN)
set(QNN_LIB_PATH                    ${QNN_SDK_PATH}/lib/aarch64-android)

include_directories(${QNN_INC_PATH})
include_directories(../../)  # ggml.h

set(SOURCE_FILES
        ../../ggml.c
        ../../ggml-alloc.c
        ../../ggml-backend.c
        ../../ggml-quants.c
        ../../ggml-qnn.cpp
        test-qnn-ops.cpp
)


message("QNN_SDK_PATH         : ${QNN_SDK_PATH}")
message("QNN_INC_PATH         : ${QNN_INC_PATH}")
message("QNN_LIB_PATH         : ${QNN_LIB_PATH}")

add_definitions(-D__ARM_NEON)
add_definitions(-DGGML_USE_QNN)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
add_definitions(-DNDEBUG)
add_definitions(-O3)
endif()

if (TARGET_SNAPDRAGON_8_GEN3)
# the below build optimization only verified and works well on Qualcomm SM8650-AB Snapdragon 8 Gen 3
add_definitions(-march=armv8.7-a)
add_definitions(-mcpu=cortex-x1)
add_definitions(-mtune=cortex-x1)

else()
# the below build optimization might be works well on ALL mainstream Android phone based on Qualcomm mobile SoC
add_definitions(-mcpu=cortex-a72)

endif()

add_compile_options("-Wall" "-Wno-sign-compare")

find_library(LOG_LIB log)

link_libraries(${LOG_LIB} android)

add_executable(${TARGET_NAME}
    ${SOURCE_FILES}
)
