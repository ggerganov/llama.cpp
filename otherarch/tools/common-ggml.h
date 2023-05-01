#pragma once

#include "ggml.h"

#include <map>
#include <fstream>
#include <vector>
#include <string>


void ggml_print_ftypes(FILE * fp = stderr);

enum ggml_ftype ggml_parse_ftype(const char * str);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);