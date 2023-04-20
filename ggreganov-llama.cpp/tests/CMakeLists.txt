function(llama_add_test source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    target_link_libraries(${TEST_TARGET} PRIVATE llama)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
endfunction()

# llama_add_test(test-double-float.c) # SLOW
llama_add_test(test-quantize.c)
llama_add_test(test-tokenizer-0.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab.bin)
