set(TARGET mulmat-tune)
add_executable(${TARGET} mulmat-tune.cpp)

if (XCODE OR MSVC)
  set(MULMAT_TUNE_LIBS ggml)
else()
  set(MULMAT_TUNE_LIBS ggml m)
endif()

target_link_libraries(${TARGET} PRIVATE ${MULMAT_TUNE_LIBS} ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
