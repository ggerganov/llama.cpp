#pragma once

#include <string>
#include <vector>
#include <fstream>
#include "examples/server/json.hpp"

static nlohmann::json get_json(const char* file_name) noexcept {
    try {
        printf("Opening a json file %s\n", file_name);
        std::ifstream jstream(file_name);
        return nlohmann::json::parse(jstream);
    }
    catch (const std::exception& ex) {
        fprintf(stderr, "%s\n", ex.what());
    }

    return {};
}

struct args_struct {
    size_t argc;
    char** argv;
    size_t elements_count     = 0;
    size_t index              = 0;

    std::vector<char>   arg_chars = {};
    std::vector<size_t> arg_idxs  = {};
    std::vector<char*>  arg_ptrs  = {};

    args_struct() = default;

    args_struct(char* file_name) {
        createParams(file_name);
    }

    ~args_struct() = default;

    void reset() noexcept {
        elements_count = 0;
        index = 0;

        arg_chars.clear();
        arg_idxs.clear();

        reset_args();
    }

    void reset_args() noexcept {
        arg_ptrs.clear();
        argc = 0;
        argv = nullptr;
    }

    void add(const std::string& data) {
        // resetting previous args
        reset_args();

        arg_idxs.emplace_back(index);
        for(const auto& character : data) {
            arg_chars.emplace_back(character);
            ++index;
        }

        arg_chars.emplace_back('\0');
        ++index;
        ++elements_count;
    }

    void get_args() {
        reset_args();
        if(elements_count) {
            arg_ptrs.reserve(elements_count);

            for(const auto& index : arg_idxs) {
                arg_ptrs.emplace_back(&arg_chars[index]);
            }
        } else {
            arg_ptrs.emplace_back(nullptr);
        }

        argc = elements_count;
        argv = &arg_ptrs[0];
    }

    void createParams(char* file_name) {
        reset(); // starting over
        nlohmann::json file_config = get_json(file_name);
        if (!file_config.empty()) { // ensures no unnecessary work
            add(file_name);
            for (auto& p : file_config.items()) {
                // only use strings, numbers and booleans for switches
                if (p.value().is_string() || p.value().is_number() || p.value().is_boolean()) {
                    add(p.key());

                    if (!p.value().is_boolean()) {
                        std::string param_value;
                        if (p.value().is_string()) {
                            param_value = p.value().get<std::string>();
                        } else if (p.value().is_number()) {
                            param_value = std::to_string(p.value().get<float>()); // works for int values too
                        }
                        add(param_value);
                    }
                }
            }
            get_args();
        }
    }
};
