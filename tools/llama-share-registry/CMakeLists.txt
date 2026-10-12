set(TARGET llama-share-registry)
add_executable(${TARGET} llama-share-registry.cpp)
# llama-share.h pulls in ggml.h / ggml-backend.h for the buffer types
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ../../src ../../include ../../ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()