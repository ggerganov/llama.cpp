message(STATUS "Using RPC backend")

set(GGML_CDEF_PUBLIC ${GGML_CDEF_PUBLIC} GGML_USE_RPC PARENT_SCOPE)

add_library(ggml-rpc
            ggml-rpc.cpp)

target_link_libraries(ggml-rpc PRIVATE ggml-base)
target_include_directories(ggml-rpc PRIVATE . ..)

if (WIN32)
    target_link_libraries(ggml-rpc PRIVATE ws2_32)
endif()
