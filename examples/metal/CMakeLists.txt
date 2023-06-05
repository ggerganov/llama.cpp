set(TEST_TARGET metal)
add_executable(${TEST_TARGET} metal.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
