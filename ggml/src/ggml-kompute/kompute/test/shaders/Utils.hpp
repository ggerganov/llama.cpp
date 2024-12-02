// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

/**
 * Compile a single glslang source from string value. This is only meant
 * to be used for testing as it's non threadsafe, and it had to be removed
 * from the glslang dependency and now can only run the CLI directly due to
 * license issues: see https://github.com/KomputeProject/kompute/pull/235
 *
 * @param source An individual raw glsl shader in string format
 * @return The compiled SPIR-V binary in unsigned int32 format
 */
std::vector<uint32_t>
compileSource(const std::string& source);
