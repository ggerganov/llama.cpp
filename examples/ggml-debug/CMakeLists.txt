set(TARGET ggml-debug)
add_executable(${TARGET} ggml-debug.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# define tests
enable_testing()

add_test(NAME test-ggml-debug COMMAND ggml-debug --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf --model stories260K.gguf --prompt hello --seed 42)