#pragma once

#include <string>

struct llama_model_quantize_params;

void llama_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params);
