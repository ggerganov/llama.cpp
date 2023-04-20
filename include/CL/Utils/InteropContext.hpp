#pragma once

#include "OpenCLUtilsCpp_Export.h"
#include <CL/Utils/Error.hpp>

#include <CL/opencl.hpp>

namespace cl {
namespace util {
    vector<cl_context_properties>
        UTILSCPP_EXPORT get_interop_context_properties(const cl::Device& plat,
                                                       cl_int* error = nullptr);

    Context UTILSCPP_EXPORT get_interop_context(int plat_id, int dev_id,
                                                cl_device_type type,
                                                cl_int* error = nullptr);
}
}
