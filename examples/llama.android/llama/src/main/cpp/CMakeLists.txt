# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE  Release)
# Declares the project name
project("llama-android")

# Enable FetchContent module
include(FetchContent)

FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# Declare llama.cpp repository
FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/NexaAI/ggml-project-apollo.git
        GIT_TAG main
        SOURCE_SUBDIR llama.cpp_74d73dc
)

# Declare llama.cpp repository
FetchContent_Declare(
        llava
        GIT_REPOSITORY https://github.com/NexaAI/ggml-project-apollo.git
        GIT_TAG main
        SOURCE_SUBDIR llama.cpp_74d73dc/examples/llava
)

FetchContent_Declare(
        omni_vlm
        GIT_REPOSITORY https://github.com/NexaAI/ggml-project-apollo.git
        GIT_TAG main
        SOURCE_SUBDIR llama.cpp_74d73dc/examples/omni-vlm
)

# Make the content available
FetchContent_MakeAvailable(llama llava omni_vlm)

# Create the main library
add_library(${CMAKE_PROJECT_NAME} SHARED
        llama-android.cpp
        common.cpp
        llava-android.cpp
)


# Link the required libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
        nlohmann_json
        llama
        common
        android
        log
        llava
)


add_library(omni-android SHARED
        llama-android.cpp
        common.cpp
        omni-android.cpp
)


# Link the required libraries
target_link_libraries(omni-android
        nlohmann_json
        llama
        common
        android
        log
        omni_vlm
)

