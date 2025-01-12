#pragma once

#include "common.h"

#include <set>
#include <string>
#include <vector>

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    std::vector<const char *> args;
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sparam = false; // is current arg a sampling param?
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;

    common_arg(
        const std::initializer_list<const char *> & args_,
        const char * value_hint_,
        const std::string & help_,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args_), value_hint(value_hint_), help(help_), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args_,
        const char * value_hint_,
        const std::string & help_,
        void (*handler)(common_params & params, int)
    ) : args(args_), value_hint(value_hint_), help(help_), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args_,
        const std::string & help_,
        void (*handler)(common_params & params)
    ) : args(args_), help(help_), handler_void(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args_,
        const char * value_hint_,
        const char * value_hint_2_,
        const std::string & help_,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args_), value_hint(value_hint_), value_hint_2(value_hint_2_), help(help_), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> vals);
    common_arg & set_excludes(std::initializer_list<enum llama_example> vals);
    common_arg & set_env(const char * val);
    common_arg & set_sparam();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;
};

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params_) : params(params_) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// function to be used by test-arg-parser
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);
