# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(main)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (LLAMA_BUILD_SERVER)
    endif()
endif()
