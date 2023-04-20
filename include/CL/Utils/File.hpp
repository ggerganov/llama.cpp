#pragma once

// OpenCL SDK includes
#include <CL/Utils/Utils.hpp>

// STL includes
#include <fstream>
#include <string>

namespace cl {
namespace util {
    // Scott Meyers, Effective STL, Addison-Wesley Professional, 2001, Item 29
    // with error handling
    UTILSCPP_EXPORT
    std::string read_text_file(const char* const filename, cl_int* const error)
    {
        std::ifstream in(filename);
        if (in.good())
        {
            try
            {
                std::string red((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
                if (in.good() && in.eof())
                {
                    if (error != nullptr) *error = CL_SUCCESS;
                    return red;
                }
                else
                {
                    detail::errHandler(CL_UTIL_FILE_OPERATION_ERROR, error,
                                       "File read error!");
                    return std::string();
                }
            } catch (std::bad_alloc& ex)
            {
                detail::errHandler(CL_OUT_OF_RESOURCES, error,
                                   "Bad allocation!");
                return std::string();
            }
        }
        else
        {
            detail::errHandler(CL_INVALID_VALUE, error, "No file!");
            return std::string();
        }
    }
}
}
