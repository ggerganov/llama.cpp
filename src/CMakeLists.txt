llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

# TODO: This is intrusive. We intend to remove SMALL_ALLOC path once the we fully
# migrate to the non SMALL_ALLOC path.
if (GGML_OPENCL)
    add_compile_definitions(GGML_USE_OPENCL)
    if (GGML_OPENCL_SMALL_ALLOC)
        add_compile_definitions(GGML_OPENCL_SMALL_ALLOC)
    endif ()
endif ()

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
