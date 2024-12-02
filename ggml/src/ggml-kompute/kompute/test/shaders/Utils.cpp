// SPDX-License-Identifier: Apache-2.0

#include "Utils.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::vector<uint32_t>
compileSource(const std::string& source)
{
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(
          std::string(
            "glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
            .c_str())) {
        throw std::runtime_error("Error running glslangValidator command");
    }
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(
      buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return { reinterpret_cast<uint32_t*>(buffer.data()),
             reinterpret_cast<uint32_t*>(buffer.data() + buffer.size()) };
}
