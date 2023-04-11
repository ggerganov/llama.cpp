#pragma once

#include "OpenCLUtilsCpp_Export.h"
#include <CL/Utils/Error.hpp>

#include <CL/opencl.hpp>

namespace cl {
namespace util {
    bool UTILSCPP_EXPORT opencl_c_version_contains(
        const cl::Device& device, const cl::string& version_fragment);

    bool UTILSCPP_EXPORT supports_extension(const cl::Device& device,
                                            const cl::string& extension);

#ifdef CL_VERSION_3_0
    bool UTILSCPP_EXPORT supports_feature(const cl::Device& device,
                                          const cl::string& feature_name);
#endif
}
}
