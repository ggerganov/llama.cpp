function(llama_build_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE llama common)
endfunction()

function(llama_test_executable name source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    # add_executable(${TEST_TARGET} ${source})
    # install(TARGETS ${TEST_TARGET} RUNTIME)
    # target_link_libraries(${TEST_TARGET} PRIVATE llama)
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
#llama_test_executable (test-tokenizer-0-falcon test-tokenizer-0-falcon.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_build_executable(test-tokenizer-1.cpp)
# test-tokenizer-1 requires a BPE vocab. re-enable when we have one.
#llama_test_executable (test-tokenizer-1.llama test-tokenizer-1.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
#llama_test_executable(test-tokenizer-1.aquila test-tokenizer-1.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
llama_build_and_test_executable(test-grammar-parser.cpp)
llama_build_and_test_executable(test-llama-grammar.cpp)
llama_build_and_test_executable(test-grad0.cpp) # SLOW
# llama_build_and_test_executable(test-opt.cpp) # SLOW
