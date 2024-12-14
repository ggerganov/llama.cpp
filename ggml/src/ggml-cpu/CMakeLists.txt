function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/ggml-cpu-aarch64.cpp
        ggml-cpu/ggml-cpu-aarch64.h
        ggml-cpu/ggml-cpu-hbm.cpp
        ggml-cpu/ggml-cpu-hbm.h
        ggml-cpu/ggml-cpu-quants.c
        ggml-cpu/ggml-cpu-quants.h
        ggml-cpu/ggml-cpu-traits.cpp
        ggml-cpu/ggml-cpu-traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/ggml-cpu-impl.h
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            message(WARNING "OpenMP not found")
        endif()
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
        CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
        (NOT CMAKE_OSX_ARCHITECTURES      AND
        NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))

        message(STATUS "ARM detected")

        if (MSVC)
            list(APPEND ARCH_DEFINITIONS __aarch64__) # MSVC defines _M_ARM64 instead
            list(APPEND ARCH_DEFINITIONS __ARM_NEON)
            list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_FMA)

            set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
            string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")

            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
            if (GGML_COMPILER_SUPPORT_DOTPROD)
                list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_DOTPROD)

                message(STATUS "ARM feature DOTPROD enabled")
            endif ()

            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmmlaq_f32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)

            if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
                list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_MATMUL_INT8)

                message(STATUS "ARM feature MATMUL_INT8 enabled")
            endif ()

            check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
                list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

                message(STATUS "ARM feature FP16_VECTOR_ARITHMETIC enabled")
            endif ()

            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
        elseif (APPLE)
            if (GGML_NATIVE)
                set(USER_PROVIDED_MARCH FALSE)
                foreach(flag_var IN ITEMS CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_REQUIRED_FLAGS)
                    if ("${${flag_var}}" MATCHES "-march=[a-zA-Z0-9+._-]+")
                        set(USER_PROVIDED_MARCH TRUE)
                        break()
                    endif()
                endforeach()

                if (NOT USER_PROVIDED_MARCH)
                    set(MARCH_FLAGS "-march=armv8.2a")

                    check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
                    if (GGML_COMPILER_SUPPORT_DOTPROD)
                        set(MARCH_FLAGS "${MARCH_FLAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_DOTPROD)

                        message(STATUS "ARM feature DOTPROD enabled")
                    endif ()

                    set(TEST_I8MM_FLAGS "-march=armv8.2a+i8mm")

                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    set(CMAKE_REQUIRED_FLAGS     "${CMAKE_REQUIRED_FLAGS} ${TEST_I8MM_FLAGS}")

                    check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)
                    if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
                        set(MARCH_FLAGS "${MARCH_FLAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS __ARM_FEATURE_MATMUL_INT8)

                        message(STATUS "ARM feature MATMUL_INT8 enabled")
                    endif ()

                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

                    list(APPEND ARCH_FLAGS "${MARCH_FLAGS}")
                endif ()
            endif ()
        else()
            check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
            if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                list(APPEND ARCH_FLAGS -mfp16-format=ieee)
            endif()
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
                # Raspberry Pi 1, Zero
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
            endif()
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
                if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                    # Android armeabi-v7a
                    list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
                else()
                    # Raspberry Pi 2
                    list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
                endif()
            endif()
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
                # Android arm64-v8a
                # Raspberry Pi 3, 4, Zero 2 (32-bit)
                list(APPEND ARCH_FLAGS -mno-unaligned-access)
            endif()
            if (GGML_SVE)
                list(APPEND ARCH_FLAGS -march=armv8.6-a+sve)
            endif()
        endif()
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
            (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            else ()
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                # MSVC generates AVX512 with AVX-VNNI intrinsics even with /arch:AVX2
                #list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                list(APPEND ARCH_FLAGS -msse4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
        message(STATUS "PowerPC detected")
        execute_process(COMMAND bash -c "grep POWER10 /proc/cpuinfo | head -n 1" OUTPUT_VARIABLE POWER10_M)
        string(FIND "${POWER10_M}" "POWER10" substring_index)
        if (NOT DEFINED substring_index OR "${substring_index}" STREQUAL "")
            set(substring_index -1)
        endif()

        if (${substring_index} GREATER_EQUAL 0)
        list(APPEND ARCH_FLAGS -mcpu=power10)
        elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
        else()
            list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
            # TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
        endif()
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        message(STATUS "loongarch64 detected")

        list(APPEND ARCH_FLAGS -march=loongarch64)
        if (GGML_LASX)
            list(APPEND ARCH_FLAGS -mlasx)
        endif()
        if (GGML_LSX)
            list(APPEND ARCH_FLAGS -mlsx)
        endif()
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
        message(STATUS "RISC-V detected")
        if (GGML_RVV)
            list(APPEND ARCH_FLAGS -march=rv64gcv -mabi=lp64d)
        endif()
    else()
        message(STATUS "Unknown architecture")
    endif()

    if (GGML_CPU_AARCH64)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_AARCH64)
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})

    if (GGML_BACKEND_DL)
        # The feature detection code is compiled as a separate target so that
        # it can be built without the architecture flags
        # Since multiple variants of the CPU backend may be included in the same
        # build, using set_source_files_properties() to set the arch flags is not possible
        set(GGML_CPU_FEATS_NAME ${GGML_CPU_NAME}-feats)
        add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/cpu-feats-x86.cpp)
        target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . .. ../include)
        target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARCH_DEFINITIONS})
        target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
        set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_FEATS_NAME})
    endif()

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()
endfunction()
