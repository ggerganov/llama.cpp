cmake_minimum_required(VERSION 3.17.0)
project(kompute_godot VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

set(KOMPUTE_EXTRA_CXX_FLAGS "" CACHE STRING "Extra compile flags for Kompute, see docs for full list")

# It is necessary to pass the DEBUG or RELEASE flag accordingly to Kompute
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1 ${KOMPUTE_EXTRA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE=1 ${KOMPUTE_EXTRA_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CC_FLAGS} /W3 /MDd /Zi /EHsc /Ox /DNDEBUG /FS")

find_package(kompute REQUIRED)
find_package(Vulkan REQUIRED)

add_library(kompute_godot
    SHARED
    src/KomputeSummator.cpp
    src/KomputeGdNative.cpp)

target_include_directories(
    kompute_godot PUBLIC
    src/
    godot-cpp/include 
    godot-cpp/include/core 
    godot-cpp/include/gen 
    godot-cpp/godot_headers

)

target_link_libraries(kompute_godot
    kompute::kompute
    Vulkan::Vulkan
)

target_link_libraries(kompute_godot
    ${CMAKE_CURRENT_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.windows.release.default.lib
)

