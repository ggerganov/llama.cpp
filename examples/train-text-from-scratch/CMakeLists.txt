set(TARGET train-text-from-scratch)
add_executable(${TARGET} train-text-from-scratch.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# TODO(cebtenzzre): remove this once PR #2632 gets merged
if (NOT MSVC)
    target_compile_options(${TARGET} PRIVATE -Wno-missing-declarations)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET} PRIVATE -Wno-missing-prototypes)
endif()
