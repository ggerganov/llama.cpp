set(TARGET baby-llama)
add_executable(${TARGET} baby-llama.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

add_executable(${TARGET}-text baby-llama-text.cpp)
target_link_libraries(${TARGET}-text PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET}-text PRIVATE cxx_std_11)
