set(TEST_TARGET test-tokenizer-0)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE llama ggml utils)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab.bin)
