# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${source} get-model.cpp)
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# llama_test(test-double-float.cpp) # SLOW
llama_test(test-quantize-fns.cpp)
llama_test(test-quantize-perf.cpp)
llama_test(test-sampling.cpp)
llama_test(test-chat-template.cpp)

llama_test(test-tokenizer-0-llama.cpp  NAME test-tokenizer-0-llama            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama.gguf)
llama_test(test-tokenizer-0-falcon.cpp NAME test-tokenizer-0-falcon           ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)

llama_test(test-tokenizer-1-llama.cpp  NAME test-tokenizer-1-llama            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama.gguf)
llama_test(test-tokenizer-1-llama.cpp  NAME test-tokenizer-1-baichuan         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-falcon           ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-aquila           ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-mpt              ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-stablelm-3b-4e1t ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-stablelm-3b-4e1t.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-gpt-neox         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-refact           ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-starcoder        ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)
llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-gpt2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt2.gguf)
#llama_test(test-tokenizer-1-bpe.cpp    NAME test-tokenizer-1-bloom            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bloom.gguf) # BIG

llama_test(test-grammar-parser.cpp)
llama_test(test-llama-grammar.cpp)
llama_test(test-grammar-integration.cpp)
llama_test(test-grad0.cpp)
# llama_test(test-opt.cpp) # SLOW
llama_test(test-backend-ops.cpp)

llama_test(test-rope.cpp)

llama_test(test-model-load-cancel.cpp  LABEL "model")
llama_test(test-autorelease.cpp        LABEL "model")

llama_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)
