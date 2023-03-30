set(TARGET main_rwkv)
add_executable(${TARGET} main_rwkv.cpp)
target_link_libraries(${TARGET} PRIVATE common ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
