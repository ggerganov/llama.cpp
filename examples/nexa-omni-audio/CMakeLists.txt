# whisper

# Find the Threads package
find_package(Threads REQUIRED)

# build nexa-whisper-utils
set(WHISPER_LIB nexa-whisper-utils)
add_library(${WHISPER_LIB} OBJECT
            whisper.cpp
            )
target_link_libraries(${WHISPER_LIB} PRIVATE ggml_llama common Threads::Threads)

# build the whisper encoder
# add_executable(whisper-encode main-encode.cpp)
# target_link_libraries(whisper-encode PRIVATE ggml_llama common Threads::Threads ${WHISPER_LIB})

# build the audio projector
# add_executable(audio-projector-cli audio-projector-cli.cpp audio-projector.cpp)
# target_link_libraries(audio-projector-cli PRIVATE ggml_llama common)

# add nexa-omni-audio-lib library
set(OMNI_AUDIO_LIB nexa-omni-audio-lib)
add_library(${OMNI_AUDIO_LIB} OBJECT
            omni.cpp
            omni.h
            audio-projector.cpp
            audio-projector.h
            )
target_link_libraries(${OMNI_AUDIO_LIB} PRIVATE ggml_llama common ${WHISPER_LIB})

# build the nexa-omni-cli
add_executable(nexa-omni-cli omni-cli.cpp)
target_link_libraries(nexa-omni-cli PRIVATE ggml_llama common Threads::Threads ${WHISPER_LIB} ${OMNI_AUDIO_LIB})

# If BUILD_SHARED_LIBS is ON, also build a shared library
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries")

    set_target_properties(${WHISPER_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${OMNI_AUDIO_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    add_library(${OMNI_AUDIO_LIB}_shared SHARED $<TARGET_OBJECTS:${OMNI_AUDIO_LIB}>)
    target_link_libraries(${OMNI_AUDIO_LIB}_shared PRIVATE ggml_llama common ${WHISPER_LIB})
    # NEXA AI : must have below two lines to make Neexa SDK export the shared library to the correct location
    set_target_properties(${OMNI_AUDIO_LIB}_shared PROPERTIES
                            PUBLIC_HEADER omni.h
                            POSITION_INDEPENDENT_CODE ON
                            OUTPUT_NAME "${OMNI_AUDIO_LIB}_shared"
    )
    install(TARGETS ${OMNI_AUDIO_LIB}_shared 
    LIBRARY
    PUBLIC_HEADER DESTINATION include
    )

    # Add OMNI_AUDIO_SHARED definition when building the shared library
    target_compile_definitions(${OMNI_AUDIO_LIB}_shared PRIVATE OMNI_AUDIO_SHARED WHISPER_SHARED)

    # Ensure all symbols are exported on Windows
    if(MSVC)
        set_target_properties(${OMNI_AUDIO_LIB}_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()