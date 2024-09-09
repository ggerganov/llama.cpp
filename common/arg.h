#pragma once

#include "common.h"

#include <string>
#include <vector>
#include <set>

//
// CLI argument parsing
//

struct gpt_params;

enum llama_example {
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_MAIN,
    LLAMA_EXAMPLE_INFILL,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_LLAVA,

    LLAMA_EXAMPLE_COUNT,
};

struct llama_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::vector<const char *> args;
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    void (*handler_void)   (gpt_params & params) = nullptr;
    void (*handler_string) (gpt_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(gpt_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (gpt_params & params, int) = nullptr;

    llama_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(gpt_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    llama_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(gpt_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    llama_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(gpt_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    // support 2 values for arg
    llama_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(gpt_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    llama_arg & set_examples(std::initializer_list<enum llama_example> examples);
    llama_arg & set_env(const char * env);
    bool in_example(enum llama_example ex);
    bool get_value_from_env(std::string & output);
    bool has_value_from_env();
    std::string to_string();
};

struct llama_arg_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    gpt_params & params;
    std::vector<llama_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    llama_arg_context(gpt_params & params) : params(params) {}
};

// initialize list of options (arguments) that can be used by the current example
llama_arg_context gpt_params_parser_init(gpt_params & params, llama_example ex);
// optionally, we can provide "print_usage" to print example usage
llama_arg_context gpt_params_parser_init(gpt_params & params, llama_example ex, void(*print_usage)(int, char **));

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
bool gpt_params_parse(int argc, char ** argv, llama_arg_context & ctx_arg);
