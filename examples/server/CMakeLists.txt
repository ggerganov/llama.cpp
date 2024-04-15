set(SERVER_ASSETS
    index.html
    index.js
    completion.js
    json-schema-to-grammar.mjs
)
foreach(asset ${SERVER_ASSETS})
    # Portable equivalent of `cd public && xxd -i ${asset} ../${asset}.hpp`
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")

    string(REGEX REPLACE "\\.| |-" "_" name "${asset}")
    file(READ "${input}" data HEX)
    string(LENGTH ${data} hex_len)
    math(EXPR len "${hex_len} / 2")
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," data ${data})
    file(WRITE "${output}" "unsigned char ${name}[] = {${data}};\nunsigned int ${name}_len = ${len};\n")
endforeach()

set(TARGET server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
option(LLAMA_SERVER_SSL "Build SSL support for the server" OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_executable(${TARGET}
    server.cpp
    utils.hpp
    httplib.h
)
install(TARGETS ${TARGET} RUNTIME)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
target_link_libraries(${TARGET} PRIVATE common ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_11)
