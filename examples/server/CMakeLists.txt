set(SERVER_ASSETS
    index.html
    index.js
    completion.js
    json-schema-to-grammar.mjs
)
set(SERVER_ASSETS_HPP "")
foreach(asset ${SERVER_ASSETS})
  list(APPEND SERVER_ASSETS_HPP "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
endforeach()
foreach(asset ${SERVER_ASSETS})
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/public"
    COMMAND     xxd -i "${asset}" "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}"
    VERBATIM
  )
endforeach()
add_custom_target(generate_server_assets ALL DEPENDS ${SERVER_ASSETS_HPP})

set(TARGET server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_executable(${TARGET}
    server.cpp
    utils.hpp
    httplib.h
)
add_dependencies(${TARGET} generate_server_assets)
install(TARGETS ${TARGET} RUNTIME)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
target_link_libraries(${TARGET} PRIVATE common json-schema-to-grammar ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)
