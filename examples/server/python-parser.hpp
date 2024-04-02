#include <iostream>
#include <tree_sitter/api.h>
#include <cstring>
#include <vector>
#include <string>
#include "json.hpp" // Include the JSON library

extern "C" TSLanguage *tree_sitter_python();

using json = nlohmann::json; // Use an alias for easier access


static json parseValue(const std::string& content) {
    // Check for boolean
    if (content == "true" || content == "True") {
        return true;
    } else if (content == "false" || content == "False") {
        return false;
    }
    // Check for quoted string
    if ((content.size() >= 2 && (content.front() == '"' && content.back() == '"')) ||
        (content.size() >= 2 && (content.front() == '\'' && content.back() == '\''))) {
        return content.substr(1, content.size() - 2);
    }
    // Attempt to parse as number (int or float)
    try {
        size_t processed;
        // Try integer first
        int i = std::stoi(content, &processed);
        if (processed == content.size()) return i;
        // Then try floating point
        double d = std::stod(content, &processed);
        if (processed == content.size()) return d;
    } catch (const std::invalid_argument& e) {
        // Not a number, ignore
    } catch (const std::out_of_range& e) {
        // Number out of range, ignore
    }
    // TODO: for array, dict, object, function, should further add logic to parse them recursively.
    return content;
}


// Recursive function to parse and create JSON for the outer function calls
static void parseFunctionCalls(const TSNode& node, std::vector<json>& calls, const char* source_code, uint32_t indent = 0) {
    auto type = ts_node_type(node);

    // printf("type: %s\n", type);
    // Only interested in call_expression nodes at the outermost level
    if (strcmp(type, "call") == 0) {
        
        json call = {
            {"id", std::to_string(calls.size())},
            {"name", ""},
            {"args", json::array()},
            {"kwargs", json::object()}
        };

        TSNode functionNode = ts_node_child(node, 0); // The function name node
        TSNode argumentsNode = ts_node_child(node, 1); // The arguments node
        
        // Extract the function name
        call["name"] = std::string(source_code + ts_node_start_byte(functionNode), ts_node_end_byte(functionNode) - ts_node_start_byte(functionNode));
        
        unsigned int numArgs = ts_node_named_child_count(argumentsNode);
        for (unsigned int i = 0; i < numArgs; ++i) {
            TSNode argNode = ts_node_named_child(argumentsNode, i);
            const char* argType = ts_node_type(argNode);
            
            // Check if the argument is a positional argument or a keyword argument
            if (strcmp(argType, "argument") == 0 || strcmp(argType, "positional_arguments") == 0 || strcmp(argType, "string") == 0 || strcmp(argType, "integer") == 0 || strcmp(argType, "true") == 0 || strcmp(argType, "false") == 0) {
                std::string value = std::string(source_code + ts_node_start_byte(argNode), ts_node_end_byte(argNode) - ts_node_start_byte(argNode));
                call["args"].push_back(parseValue(value));
            } else if (strcmp(argType, "keyword_argument") == 0) {
                // Extract keyword and value for keyword arguments
                TSNode keyNode = ts_node_child(argNode, 0); // The key of the kwarg
                TSNode valueNode = ts_node_child(argNode, 2); // The value of the kwarg, 1 is the symbol `=`

                // if this is 0 then it's a string/integer/boolean, simply parse it
                // unsigned int numValueNodeChild = ts_node_named_child_count(valueNode);
                // TODO: if numValueNodeChild != 0 then it's an array/list/object?/function. Need to do something more. However for now we assume this will not happen.

                std::string key = std::string(source_code + ts_node_start_byte(keyNode), ts_node_end_byte(keyNode) - ts_node_start_byte(keyNode));
                std::string value = std::string(source_code + ts_node_start_byte(valueNode), ts_node_end_byte(valueNode) - ts_node_start_byte(valueNode));
                call["kwargs"][key] = parseValue(value);
            }
        }
        
        calls.push_back(call);
        return; // Stop recursion to only process outer function calls
    }

    // Recurse through all children for other node types
    unsigned int numChildren = ts_node_child_count(node);
    for (unsigned int i = 0; i < numChildren; ++i) {
        TSNode child = ts_node_child(node, i);
        parseFunctionCalls(child, calls, source_code, indent+1);
    }
}

static std::vector<json> parsePythonFunctionCalls(std::string source_string) {
    // Parse Python function calls from the source code and return a JSON array
    std::vector<json> calls;
    std::string delimiter = "<<functions>>";
    std::string source_code;
    printf("Parsing source_string::%s\n", source_string.c_str());
    size_t startPos = source_string.find(delimiter);
    if (startPos != std::string::npos) {
        source_code = source_string.substr(startPos + delimiter.length());
    } else {
        return calls;
    }
    TSParser *parser = ts_parser_new();
    ts_parser_set_language(parser, tree_sitter_python());
    const char* source_code_cstr = source_code.c_str();
    TSTree *tree = ts_parser_parse_string(parser, nullptr, source_code_cstr, source_code.length());
    TSNode root_node = ts_tree_root_node(tree);

    bool has_errors = ts_node_has_error(root_node);

    if (has_errors) {
        // probably a regular string
        printf("has errors\n");
        return calls;
    }

    parseFunctionCalls(root_node, calls, source_code_cstr, 0);

    ts_tree_delete(tree);
    ts_parser_delete(parser);
    printf("calls: %s\n", json(calls).dump().c_str());
    return calls;
}

