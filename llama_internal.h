// Internal header to be included by llama.cpp and tests/benchmarks only.

#ifndef LLAMA_INTERNAL_H
#define LLAMA_INTERNAL_H

#include <vector>
#include <string>
struct ggml_tensor;

std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(struct llama_context * ctx);

#endif // LLAMA_INTERNAL_H
