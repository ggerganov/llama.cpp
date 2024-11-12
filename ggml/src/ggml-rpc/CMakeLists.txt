message(STATUS "Using RPC backend")

add_library(ggml-rpc
            ggml-rpc.cpp)

target_link_libraries(ggml-rpc PRIVATE ggml-base)
target_include_directories(ggml-rpc PRIVATE . ..)

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
