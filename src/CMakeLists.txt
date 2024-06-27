# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# nvapi
if (LLAMA_NVAPI)
    add_library(nvapi nvapi.cpp nvapi.h)

    if (LINUX)
        target_link_libraries(nvapi PUBLIC dl)
    endif()
endif()

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_NVAPI)
    target_link_libraries(llama PUBLIC nvapi)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()
