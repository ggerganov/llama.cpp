if(KOMPUTE_OPT_REPO_SUBMODULE_BUILD)
    message(FATAL_ERROR "'KOMPUTE_OPT_REPO_SUBMODULE_BUILD' got replaced by 'KOMPUTE_OPT_USE_BUILT_IN_SPDLOG', 'KOMPUTE_OPT_USE_BUILT_IN_FMT', 'KOMPUTE_OPT_USE_BUILT_IN_GOOGLE_TEST', 'KOMPUTE_OPT_USE_BUILT_IN_PYBIND11' and 'KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER'. Please use them instead.")
endif()

if(KOMPUTE_OPT_BUILD_AS_SHARED_LIB)
    message(FATAL_ERROR "'KOMPUTE_OPT_BUILD_AS_SHARED_LIB' is deprecated and should not be used. Instead use the default 'BUILD_SHARED_LIBS' CMake switch.")
endif()

if(KOMPUTE_OPT_BUILD_SINGLE_HEADER)
    message(FATAL_ERROR "'KOMPUTE_OPT_BUILD_SINGLE_HEADER' is deprecated and should not be used. The single header will now always be build and can be included via '#include<kompute/kompute.h>'.")
endif()

if(KOMPUTE_OPT_ENABLE_SPDLOG)
    message(FATAL_ERROR "'KOMPUTE_OPT_ENABLE_SPDLOG' is deprecated and should not be used. It got replaced by 'KOMPUTE_OPT_LOG_LEVEL'. This option can be set to a variety of log levels (e.g. 'Off', 'Trace', 'Debug', 'Default', ...).")
endif()