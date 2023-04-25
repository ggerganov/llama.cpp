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
endif()
