cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

# Options
option(KOMPUTE_OPT_GIT_TAG "The tag of the repo to use for the example" 1344ece4ac278f9b3be3b4555ffaace7a032b91f)
option(KOMPUTE_OPT_FROM_SOURCE "Whether to build example from source or from git fetch repo" 0)

set(KOMPUTE_OPT_ANDROID_BUILD ON)
set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON)

if(KOMPUTE_OPT_FROM_SOURCE)
    add_subdirectory(../../../../../../../ ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)
else()
    include(FetchContent)
    FetchContent_Declare(kompute GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
        GIT_TAG ${KOMPUTE_OPT_GIT_TAG})
    FetchContent_MakeAvailable(kompute)
    include_directories(${kompute_SOURCE_DIR}/src/include)
endif()

# Add to the list, so CMake can later find the code to compile shaders to header files
list(APPEND CMAKE_PREFIX_PATH "${kompute_SOURCE_DIR}/cmake")
add_subdirectory(shader)

add_library(kompute-jni SHARED KomputeJniNative.cpp
                               KomputeModelML.cpp
                               KomputeModelML.hpp)

target_link_libraries(kompute-jni PRIVATE kompute::kompute shader log android)

