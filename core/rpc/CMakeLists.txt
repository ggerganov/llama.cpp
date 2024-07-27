add_executable(rpc-server rpc-server.cpp)
target_link_libraries(rpc-server PRIVATE ggml llama)
