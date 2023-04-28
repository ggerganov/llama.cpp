#pragma once

#include "ggml.h"

#include <map>
#include <fstream>
#include <vector>
#include <string>

// model file types
enum ggml_ftype {
    GGML_FTYPE_UNKNOWN     = -1,
    GGML_FTYPE_ALL_F32     = 0,
    GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    GGML_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q4_3 = 6,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
    GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
};

void ggml_print_ftypes(FILE * fp = stderr);

enum ggml_ftype ggml_parse_ftype(const char * str);

// TODO: temporary
enum ggml_type ggml_ftype_to_ggml_type(const enum ggml_ftype ftype);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);