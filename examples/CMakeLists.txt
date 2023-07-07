# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# common

set(TARGET common)

add_library(${TARGET} OBJECT
    common.h
    common.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE llama)

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(main)
    add_subdirectory(quantize)
    add_subdirectory(quantize-stats)
    add_subdirectory(perplexity)
    add_subdirectory(embedding)
    add_subdirectory(save-load-state)
    add_subdirectory(benchmark)
    add_subdirectory(baby-llama)
    add_subdirectory(train-text-from-scratch)
    add_subdirectory(simple)
    add_subdirectory(embd-input)
    if (LLAMA_METAL)
        add_subdirectory(metal)
    endif()
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
endif()
