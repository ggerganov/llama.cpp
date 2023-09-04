# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

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
    add_subdirectory(convert-llama2c-to-ggml)
    add_subdirectory(simple)
    add_subdirectory(speculative)
    add_subdirectory(embd-input)
    add_subdirectory(llama-bench)
    add_subdirectory(beam-search)
    if (LLAMA_METAL)
        add_subdirectory(metal)
    endif()
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
endif()
