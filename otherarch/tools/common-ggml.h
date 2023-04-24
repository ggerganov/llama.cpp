#pragma once

#include <fstream>
#include <vector>
#include <string>

// model file types
enum ggml_mtype {
    GGML_MTYPE_ALL_F32     = 0,
    GGML_MTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    GGML_MTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    GGML_MTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
    GGML_MTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    GGML_MTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
    GGML_MTYPE_MOSTLY_Q4_3 = 6,  // except 1d tensors
};

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_mtype mtype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);