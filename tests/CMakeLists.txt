function(jarvis_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(JARVIS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED JARVIS_TEST_LABEL)
        set(JARVIS_TEST_LABEL "main")
    endif()
    if (NOT DEFINED JARVIS_TEST_WORKING_DIRECTORY)
        set(JARVIS_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED JARVIS_TEST_NAME)
        set(TEST_NAME ${JARVIS_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${JARVIS_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${JARVIS_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${JARVIS_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(jarvis_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(JARVIS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED JARVIS_TEST_LABEL)
        set(JARVIS_TEST_LABEL "main")
    endif()
    if (NOT DEFINED JARVIS_TEST_WORKING_DIRECTORY)
        set(JARVIS_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED JARVIS_TEST_NAME)
        set(TEST_TARGET ${JARVIS_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${source} get-model.cpp)
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${JARVIS_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${JARVIS_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${JARVIS_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-jarvis-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-jarvis-bpe.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-jarvis-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-jarvis-spm.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
jarvis_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

# build test-tokenizer-1-bpe target once and add many tests
add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
install(TARGETS test-tokenizer-1-bpe RUNTIME)

# TODO: disabled due to slowness
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-jarvis-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-jarvis-bpe.gguf --ignore-merges)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
#jarvis_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

# build test-tokenizer-1-spm target once and add many tests
add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
target_link_libraries(test-tokenizer-1-spm PRIVATE common)
install(TARGETS test-tokenizer-1-spm RUNTIME)

jarvis_test(test-tokenizer-1-spm  NAME test-tokenizer-1-jarvis-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-jarvis-spm.gguf)
#jarvis_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

# jarvis_target_and_test(test-double-float.cpp) # SLOW
jarvis_target_and_test(test-log.cpp)
jarvis_target_and_test(test-arg-parser.cpp)
jarvis_target_and_test(test-quantize-fns.cpp)
jarvis_target_and_test(test-quantize-perf.cpp)
jarvis_target_and_test(test-sampling.cpp)
jarvis_target_and_test(test-chat-template.cpp)

jarvis_target_and_test(test-grammar-parser.cpp)
jarvis_target_and_test(test-jarvis-grammar.cpp)
jarvis_target_and_test(test-grammar-integration.cpp)
jarvis_target_and_test(test-grad0.cpp)
jarvis_target_and_test(test-barrier.cpp)
# jarvis_target_and_test(test-opt.cpp) # SLOW
jarvis_target_and_test(test-backend-ops.cpp)

jarvis_target_and_test(test-rope.cpp)

jarvis_target_and_test(test-model-load-cancel.cpp  LABEL "model")
jarvis_target_and_test(test-autorelease.cpp        LABEL "model")

# TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    jarvis_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE jarvis)
