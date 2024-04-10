set(TARGET ggml-debug)
add_executable(${TARGET} ggml-debug.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

if (NOT DEFINED LLAMA_TEST_LABEL)
    set(LLAMA_TEST_LABEL "main")
endif()

set(TEST_TARGET test-ggml-debug)
add_test(NAME ${TEST_TARGET} COMMAND ggml-debug --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf --model stories260K.gguf --prompt hello --seed 42)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
