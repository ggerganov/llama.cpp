if (APPLE)
    set(TEST_TARGET mtl)
    add_executable(${TEST_TARGET} mtl.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml)
endif()

