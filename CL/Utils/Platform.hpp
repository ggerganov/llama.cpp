#pragma once

#include "OpenCLUtilsCpp_Export.h"
#include <CL/Utils/Error.hpp>

#include <CL/opencl.hpp>

namespace cl {
namespace util {
    bool UTILSCPP_EXPORT supports_extension(const cl::Platform& platform,
                                            const cl::string& extension);

    bool UTILSCPP_EXPORT platform_version_contains(
        const cl::Platform& platform, const cl::string& version_fragment);
}
}
