set(TARGET falcon_main)
add_executable(${TARGET} falcon_main.cpp)
target_link_libraries(${TARGET} PRIVATE falcon_common libfalcon ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

