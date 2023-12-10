set(TARGET main)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} libnode.so "/usr/lib/x86_64-linux-gnu/libzstd.a" "/usr/lib/x86_64-linux-gnu/libzstd.so")
target_compile_features(${TARGET} PRIVATE cxx_std_20)

