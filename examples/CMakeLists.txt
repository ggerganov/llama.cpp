# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(baby-llama)
    add_subdirectory(batched)
    add_subdirectory(batched-bench)
    add_subdirectory(beam-search)
    add_subdirectory(benchmark)
    add_subdirectory(convert-llama2c-to-ggml)
    add_subdirectory(embedding)
    add_subdirectory(finetune)
    add_subdirectory(infill)
    add_subdirectory(llama-bench)
    add_subdirectory(llava)
    if (LLAMA_SYCL)
        add_subdirectory(sycl)
    endif()
    add_subdirectory(main)
    add_subdirectory(tokenize)
    add_subdirectory(parallel)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    add_subdirectory(quantize-stats)
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(passkey)
    add_subdirectory(speculative)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(train-text-from-scratch)
    add_subdirectory(imatrix)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
    add_subdirectory(export-lora)
endif()
