#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <vector>
#include <numeric>

std::vector<std::string> splitString(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    while ((end = str.find(delimiter, start)) != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}

std::unordered_map<std::string, std::vector<std::string>> extractParameters() {
    std::ifstream file("common/common.h");
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    std::unordered_map<std::string, std::vector<std::string>> parameters;
    parameters["logit_bias"] = {"logit_bias", "0", "//", "way", "to", "alter", "prob", "of", "word", "being", "chosen"};

        bool inside = false;
    for (const std::string& line : lines) {
        std::vector<std::string> nonWhitespaceElements = splitString(line, " ");
        std::cout << "nwe = \033[33m";
        for (const std::string& element : nonWhitespaceElements) {
            std::cout << element << " ";
        }
        std::cout << "\033[0m" << std::endl;

        if (!nonWhitespaceElements.empty() && nonWhitespaceElements[0] == "struct") {
            inside = true;
        }

        if (nonWhitespaceElements.size() > 2 && inside) {
            // Note: cannot use nwe[0] because types do not generate unique keys and so overwrite
            // Here we deliberately add back the key so we can manually change it when it is different
            parameters[nonWhitespaceElements[1]] = nonWhitespaceElements;

            // Remove spurious entry caused by eccentric status of logit_bias
            if (parameters.count("float>") && parameters["float>"][1] == "logit_bias") {
                parameters.erase("float>");
            }
        }

        // Terminate the harvest
        if (nonWhitespaceElements.size() > 2 && nonWhitespaceElements[1] == "infill") {
            inside = false;
            break;
        }
    }

    for (const auto& pair : parameters) {
        const std::string& key = pair.first;
        const std::vector<std::string>& value = pair.second;
        std::cout << "key: " << std::left << std::setw(20) << key << "; values: ";
        for (const std::string& element : value) {
            std::cout << element << " ";
        }
        std::cout << std::endl;

        std::string concatenatedElement = "";
        for (std::size_t i = 0; i < value.size(); i++) {
            if (value[i] == "//") {
                concatenatedElement = std::accumulate(value.begin() + i, value.end(), std::string(""));
                // break;
            }
        }

        std::cout << std::string(10, ' ');
        std::cout << "parameter: \033[32m" << std::right << std::setw(40) << key << " \033[34mdefault: \033[30m"
                  << std::right << std::setw(5) << value[1] << " \033[34mcomment: \033[33m"
                  << std::left << std::setw(80) << concatenatedElement << "\033[0m" << std::endl;
    }

    return parameters;
}

// everything above is called from here as 'extractParameters()'
int main() {
    std::unordered_map<std::string, std::vector<std::string>> parameters = extractParameters();
    return 0;
}
