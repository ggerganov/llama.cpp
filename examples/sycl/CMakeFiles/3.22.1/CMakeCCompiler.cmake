set(CMAKE_C_COMPILER "/opt/intel/oneapi/compiler/2024.0/bin/icx")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "IntelLLVM")
set(CMAKE_C_COMPILER_VERSION "2024.0.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_C_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert;c_std_17")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "c_std_17")
set(CMAKE_C23_COMPILE_FEATURES "")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "GNU")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "4.2.1")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_C_COMPILER_AR "")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)

set(CMAKE_C_COMPILER_ENV_VAR "CC")

set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/intel/oneapi/tbb/2021.11/include;/opt/intel/oneapi/mpi/2021.11/include;/opt/intel/oneapi/mkl/2024.0/include;/opt/intel/oneapi/ippcp/2021.9/include;/opt/intel/oneapi/ipp/2021.10/include;/opt/intel/oneapi/dpl/2022.3/include;/opt/intel/oneapi/dpcpp-ct/2024.0/include;/opt/intel/oneapi/dnnl/2024.0/include;/opt/intel/oneapi/dev-utilities/2024.0/include;/opt/intel/oneapi/dal/2024.0/include/dal;/opt/intel/oneapi/compiler/2024.0/opt/oclfpga/include;/opt/intel/oneapi/ccl/2021.11/include;/opt/intel/oneapi/compiler/2024.0/opt/compiler/include;/opt/intel/oneapi/compiler/2024.0/lib/clang/17/include;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "svml;irng;imf;m;gcc;gcc_s;irc;dl;gcc;gcc_s;c;gcc;gcc_s;irc_s")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/intel/oneapi/compiler/2024.0/lib;/opt/intel/oneapi/compiler/2024.0/lib/clang/17/lib/x86_64-unknown-linux-gnu;/usr/lib/gcc/x86_64-linux-gnu/11;/usr/lib64;/lib/x86_64-linux-gnu;/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib;/opt/intel/oneapi/compiler/2024.0/opt/compiler/lib;/lib;/opt/intel/oneapi/tbb/2021.11/lib/intel64/gcc4.8;/opt/intel/oneapi/mpi/2021.11/lib;/opt/intel/oneapi/mkl/2024.0/lib;/opt/intel/oneapi/ippcp/2021.9/lib;/opt/intel/oneapi/ipp/2021.10/lib;/opt/intel/oneapi/dpl/2022.3/lib;/opt/intel/oneapi/dnnl/2024.0/lib;/opt/intel/oneapi/dal/2024.0/lib;/opt/intel/oneapi/ccl/2021.11/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
