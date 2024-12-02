
C++ Build System Deep Dive
======================

The recommended approach to build the project is as out-of-source built-in the ``build`` folder. This project uses CMake as build system.

For a base build you just have to run:

.. code-block:: bash

    git clone https://github.com/KomputeProject/kompute.git
    cd kompute
    mkdir build
    cd build
    cmake ..
    cmake --build .

This by default configures without any of the extra build tasks (such as building shaders) and compiles without the optional dependencies. The table below provides more detail.

.. list-table::
   :header-rows: 1

   * - Flag
     - Description
   * - -DCMAKE_INSTALL_PREFIX="build/src/CMakefiles/Export/"
     - Enables local installation (which won't require admin privileges)
   * - -DCMAKE_TOOLCHAIN_FILE="..."
     - This is the path for your package manager if you use it such as vcpkg.
   * - -DKOMPUTE_OPT_BUILD_TESTS=ON
     - Enable if you want to build tests.
   * - -DKOMPUTE_OPT_CODE_COVERAGE=ON
     - Enable if you want code coverage.
   * - -DKOMPUTE_OPT_BUILD_DOCS=ON
     - Enable if you want to build documentation.
   * - -DKOMPUTE_OPT_INSTALL=ON
     - Enable if you want to enable installation.
   * - -DKOMPUTE_OPT_BUILD_PYTHON=ON
     - Enable if you want to build python bindings.
   * - -DKOMPUTE_OPT_ANDROID_BUILD=ON
     - Enable android compilation flags required.
   * - -DKOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS=ON
     - Explicitly disable debug layers even on debug.
   * - -DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON
     - Whether to check if your driver supports the Vulkan Header version you are linking against. This might be useful in case you build shared on a different system than you run later.
   * - -DKOMPUTE_OPT_BUILD_SHADERS=OFF
     - Rebuilds all compute shaders during compilation and does not use the already precompiled versions. Requires glslangValidator to be installed on your system.
   * - -DKOMPUTE_OPT_LOG_LEVEL="Default"
     - Internally we use Spdlog or fmt for logging, depending on the value of 'KOMPUTE_OPT_USE_SPDLOG'. The log level used can be changed here. Possible values: 'Trace', 'Debug', 'Info', 'Warn', 'Error', 'Critical', 'Off', 'Default'. If set to 'Off' logging will be deactivated completely. If set to 'Default', the log level will be set to 'Info' for release builds and 'Debug' else.
   * - -DKOMPUTE_OPT_USE_SPDLOG=ON
     - If enabled, logging via KP_LOG_<DEBUG, INFO, etc...> will happen through Spdlog instead of plan fmt.
   * - -DKOMPUTE_OPT_USE_BUILT_IN_SPDLOG=ON
     - Use the built-in version of Spdlog. Requires 'KOMPUTE_OPT_USE_SPDLOG' to be set to ON in order to have any effect.
   * - -DKOMPUTE_OPT_SPDLOG_ASYNC_MODE=OFF
     - If enabled we use the async thread mode of spdlog, but by default using sync for consistent messages.
   * - -DKOMPUTE_OPT_USE_BUILT_IN_FMT=ON
     - Use the built-in version of fmt.
   * - -DKOMPUTE_OPT_USE_BUILT_IN_GOOGLE_TEST=ON
     - Use the built-in version of GoogleTest.
   * - -DKOMPUTE_OPT_USE_BUILT_IN_PYBIND11=ON
     - Use the built-in version of pybind11.
   * - -DKOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER=OFF
     - Use the built-in version of Vulkan Headers. This could be helpful in case your system Vulkan Headers are too new for your driver. If you set this to OFF, please make sure your system Vulkan Headers are supported by your driver.
   * - -DKOMPUTE_OPT_BUILT_IN_VULKAN_HEADER_TAG="v1.2.203"
     - The git tag used for the built-in Vulkan Headers when 'KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADER' is enabled. A list of tags can be found here: https://github.com/KhronosGroup/Vulkan-Headers/tags

Compile Flags
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Flag
     - Description
   * - KOMPUTE_CREATE_PIPELINE_RESULT_VALUE
     - Ensure the return value of createPipeline is processed as ResultValue instead of Result
   
Other CMake Flags
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Flag
     - Description
   * - -DPYTHON_INCLUDE_DIR
     - Path to where Python.h is found, used for specifying installation of Python (see `this PR <https://github.com/KomputeProject/kompute/pull/222>`_)
   * - -DPYTHON_LIBRARY
     - Path to the Python library, used for specifying installation of Python (see `this PR <https://github.com/KomputeProject/kompute/pull/222>`_)
   * - -DCMAKE_OSX_ARCHITECTURES
     - Specifies the target architecture for Apple platforms (see `this issue <https://github.com/KomputeProject/kompute/issues/223>`_)

Dependencies
^^^^^^^^^^^^

Given Kompute is expected to be used across a broad range of architectures and hardware, it will be important to make sure we are able to minimise dependencies. 

Required dependencies
~~~~~~~~~~~~~~~~~~~~~

The only required dependency in the build is the Vulkan SDK. More specifically, the header files vulkan.h and vulkan.hpp, which are both part of the Vulkan SDK. If you haven't installed the Vulkan SDK yet, you can `download it here <https://vulkan.lunarg.com/>`_.
