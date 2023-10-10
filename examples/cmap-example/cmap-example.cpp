// example of a C/C++ equivalent data structure to the python dict in readcommonh.py

#include <map>
#include <list>
#include <string>
#include <bitset>
#include <vector>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <regex>
// there may be good reasons not to sort the parameters, but here we use map
#include <map>
#include <numeric>

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    bool inside_tags = false;  // flag to track if we are inside "<>"

    while ((end = str.find(delimiter, start)) != std::string::npos) {
        std::string token = str.substr(start, end - start);

        // if (!token.empty()) { // Add condition to exclude empty substrings
        //    tokens.push_back(token);

        if (!inside_tags && !token.empty()) { // Add condition to exclude empty substrings and if not inside "<>"
            tokens.push_back(token);
        }
        // deal with cases where the split character occurs inside <>
        // Update inside_tags flag based on "<>"
        size_t open_tag_pos = str.find("<", start);
        size_t close_tag_pos = str.find(">", start);
        if (open_tag_pos != std::string::npos && close_tag_pos != std::string::npos && open_tag_pos < end) {
            inside_tags = true;
        } else if (close_tag_pos != std::string::npos && close_tag_pos < end) {
            inside_tags = false;
        }
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

void print_parameters(const std::map<std::string, std::vector<std::string>>& parameters) {
        for (const auto& pair : parameters) {
            const std::string& key = pair.first;
            const std::vector<std::string>& value = pair.second; // usually has multiple elements
            printf("key: %25s: values: ", key.c_str());
            for (const std::string& element : value) {
                printf("%s ", element.c_str());
            }
            printf("\n");
    }
}

std::map<std::string, std::vector<std::string>> extract_parameters() {
    std::ifstream file("common/common.h");
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    std::map<std::string, std::vector<std::string>> parameters;
    // fix up failure to match logit_bias; may also need to add lora_adapter; now dealt with and ready for deletion
    // parameters["logit_bias"] = {"std::unordered_map<llama_token, float>" "logit_bias", "=", "0", "//", "way", "to", "alter", "prob", "of", "word", "being", "chosen"};
    // parameters["lora_adapter"] = {"std::vector<std::tuple<std::string, float>>", "lora_adapter", "=", "", "//", "lora", "adapter", "path", "with", "user-defined", "scale"};

    // are we inside gpt_params?
    // this for loop finds all the params inside struct gpt-params
    bool inside = false;
    for (const std::string& line : lines) {
        std::vector<std::string> nws_elements = split_string(line, " ");
        printf("nwe = ");
        for (const std::string& element : nws_elements) {
            printf("%s ", element.c_str());
        }
        printf("\n");

        if (!nws_elements.empty() && nws_elements[0] == "struct" && nws_elements[1] == "gpt_params") {
            inside = true;
        }

        if (nws_elements.size() > 2 && inside) {
            // cannot use nwe[0] as key because types do not generate unique keys and so overwrite
            // Here we deliberately add back the key so we can manually change it when it is different (remove eventually)
            // parameters[nws_elements[1]] = nws_elements;
            std::vector<std::string> copy = nws_elements; // Create a copy of nws_elements
            parameters[nws_elements[1]] = copy; // Assign the copy to parameters

            // Remove spurious entry caused by eccentric status of logit_bias
            if (parameters.count("float>") && parameters["float>"][2] == "logit_bias;") {
                parameters.erase("float>");
            }
            // Remove spurious entry caused by eccentric status of lora_adapter
            if (parameters.count("float>>") && parameters["float>>"][2] == "lora_adapter;") {
                parameters.erase("float>>");
            }
        }

        // Terminate the harvest; TODO: not robust; need better terminator; this just a crude hack for now
        if (nws_elements.size() > 2 && nws_elements[2] == "infill") {
            inside = false;
            break;
            }
        }
    // now display them (unnecessary operationally; here for development)
    print_parameters(parameters);

    // return the results (will eventually become a void function)
    return parameters;
}

int main() {

    // process the code inserted to replicate readcommonh.py
    // this does not produce output but here is forced; it just collects the output into parameters and returns 0
    std::map<std::string, std::vector<std::string>> parameters = extract_parameters();
    print_parameters(parameters);

    return 0;
}
