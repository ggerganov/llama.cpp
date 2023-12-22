function(llama_build_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE llama common)
endfunction()

function(llama_test_executable name source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
endfunction()

function(llama_build_and_test_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE llama common)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
endfunction()

# llama_build_and_test_executable(test-double-float.cpp) # SLOW
llama_build_and_test_executable(test-quantize-fns.cpp)
llama_build_and_test_executable(test-quantize-perf.cpp)
llama_build_and_test_executable(test-sampling.cpp)

llama_build_executable(test-tokenizer-0-llama.cpp)
llama_test_executable (test-tokenizer-0-llama test-tokenizer-0-llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama.gguf)

llama_build_executable(test-tokenizer-0-falcon.cpp)
llama_test_executable (test-tokenizer-0-falcon test-tokenizer-0-falcon.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)

llama_build_executable(test-tokenizer-1-llama.cpp)
llama_test_executable (test-tokenizer-1-llama    test-tokenizer-1-llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama.gguf)
llama_test_executable (test-tokenizer-1-baichuan test-tokenizer-1-llama.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

llama_build_executable(test-tokenizer-1-bpe.cpp)
llama_test_executable (test-tokenizer-1-falcon           test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test_executable (test-tokenizer-1-aquila           test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
llama_test_executable (test-tokenizer-1-mpt              test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test_executable (test-tokenizer-1-stablelm-3b-4e1t test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-stablelm-3b-4e1t.gguf)
llama_test_executable (test-tokenizer-1-gpt-neox         test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
llama_test_executable (test-tokenizer-1-refact           test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test_executable (test-tokenizer-1-starcoder        test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)
# llama_test_executable (test-tokenizer-1-bloom test-tokenizer-1-bpe.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bloom.gguf) # BIG

llama_build_and_test_executable(test-grammar-parser.cpp)
llama_build_and_test_executable(test-llama-grammar.cpp)
llama_build_and_test_executable(test-grad0.cpp)
# llama_build_and_test_executable(test-opt.cpp) # SLOW
llama_build_and_test_executable(test-backend-ops.cpp)

llama_build_and_test_executable(test-rope.cpp)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)
