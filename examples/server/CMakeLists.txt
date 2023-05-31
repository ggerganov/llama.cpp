set(TARGET server)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET} server.cpp json.hpp httplib.h)
target_compile_definitions(${TARGET} PRIVATE
    $<$<CONFIG:Debug>:
        CPPHTTPLIB_NO_EXCEPTIONS=1
    >
)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()
