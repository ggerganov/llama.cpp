if (NOT GGML_ANALOG)
    return()
endif()

set(TARGET llama-analog-bench)

add_executable(${TARGET} analog-bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
