#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common.h"
#include "llama-cpp.h"

GGML_ATTRIBUTE_FORMAT(1, 2)
int printe(const char * fmt, ...);

struct Opt {
    int init(int argc, const char ** argv);

    // Public members
    llama_context_params ctx_params;
    llama_model_params   model_params;
    std::string          model_;
    std::string          user;
    int                  context_size = -1, ngl = -1;
    float                temperature = -1;
    bool                 verbose     = false;

    int   context_size_default = -1, ngl_default = -1;
    float temperature_default = -1;
    bool  help                = false;

    bool parse_flag(const char ** argv, int i, const char * short_opt, const char * long_opt);
    int  handle_option_with_value(int argc, const char ** argv, int & i, int & option_value);
    int  handle_option_with_value(int argc, const char ** argv, int & i, float & option_value);
    int  parse(int argc, const char ** argv);
    void print_help() const;
};
