#ifndef LLAMA_INTERNAL_H
#define LLAMA_INTERNAL_H

// Internal functions exposed for tests and benchmarks

#include "ggml.h"

#include <string>
#include <unordered_map>

std::unordered_map<std::string, struct ggml_tensor *>& llama_internal_get_tensor_map(struct llama_context * ctx);

#endif
