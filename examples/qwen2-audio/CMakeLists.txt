# whisper

# Find the Threads package
find_package(Threads REQUIRED)

# build nexa-whisper-utils
set(WHISPER_LIB nexa-whisper-utils-qwen2)
add_library(${WHISPER_LIB} OBJECT
            whisper.cpp
            )
target_link_libraries(${WHISPER_LIB} PRIVATE ggml llama common Threads::Threads)

# add nexa-qwen2-audio-lib library
set(QWEN2_AUDIO_LIB nexa-qwen2-audio-lib)
add_library(${QWEN2_AUDIO_LIB} OBJECT
            qwen2.cpp
            qwen2.h
            audio-projector.cpp
            audio-projector.h
            )
target_link_libraries(${QWEN2_AUDIO_LIB} PRIVATE ggml llama common ${WHISPER_LIB})

# build the nexa-qwen2-cli
add_executable(nexa-qwen2-cli qwen2-cli.cpp)
target_link_libraries(nexa-qwen2-cli PRIVATE ggml llama common Threads::Threads ${WHISPER_LIB} ${QWEN2_AUDIO_LIB})

# If BUILD_SHARED_LIBS is ON, also build a shared library
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries")

    set_target_properties(${WHISPER_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${QWEN2_AUDIO_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${QWEN2_AUDIO_LIB}_shared SHARED $<TARGET_OBJECTS:${QWEN2_AUDIO_LIB}>)
    target_link_libraries(${QWEN2_AUDIO_LIB}_shared PRIVATE ggml llama common ${WHISPER_LIB})
    # NEXA AI : must have below two lines to make Neexa SDK export the shared library to the correct location
    set_target_properties(${QWEN2_AUDIO_LIB}_shared PROPERTIES
                            PUBLIC_HEADER qwen2.h
                            POSITION_INDEPENDENT_CODE ON
                            OUTPUT_NAME "${QWEN2_AUDIO_LIB}_shared"
    )
    install(TARGETS ${QWEN2_AUDIO_LIB}_shared 
    LIBRARY
    PUBLIC_HEADER DESTINATION include
    )

    # Add QWEN2_AUDIO_SHARED definition when building the shared library
    target_compile_definitions(${QWEN2_AUDIO_LIB}_shared PRIVATE QWEN2_AUDIO_SHARED WHISPER_SHARED)

    # Ensure all symbols are exported on Windows
    if(MSVC)
        set_target_properties(${QWEN2_AUDIO_LIB}_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()
