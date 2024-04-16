#pragma once

#include "llama.h"
#include "common.h"

#include "json.hpp"
#include "python-parser.hpp"

#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <unordered_map>
#include <algorithm>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

extern bool server_verbose;
extern bool server_log_json;

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERB", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(  MSG, ...) server_log("ERR",  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARN", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(   MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value) {
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

static inline void server_log(const char *level, const char *function, int line, const char *message, const nlohmann::ordered_json &extra) {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = nlohmann::ordered_json{
        {"tid",       ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    if (server_log_json) {
        log.merge_patch( {
            {"level",    level},
            {"function", function},
            {"line",     line},
            {"msg",      message},
        });

        if (!extra.empty()) {
            log.merge_patch(extra);
        }

        printf("%s\n", log.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    } else {
        char buf[1024];
        snprintf(buf, 1024, "%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto& el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            ss << " " << el.key() << "=" << value;
        }

        const std::string str = ss.str();
        printf("%.*s\n", (int)str.size(), str.data());
        fflush(stdout);
    }
}

//
// chat template utils
//

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
inline bool verify_custom_template(const std::string & tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    int res = llama_chat_apply_template(nullptr, tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages) {
    size_t alloc_size = 0;
    // vector holding all allocated string to be passed to llama_chat_apply_template
    std::vector<std::string> str(messages.size() * 2);
    std::vector<llama_chat_message> chat(messages.size());

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & curr_msg = messages[i];
        str[i*2 + 0]    = json_value(curr_msg, "role",    std::string(""));
        str[i*2 + 1]    = json_value(curr_msg, "content", std::string(""));
        alloc_size     += str[i*2 + 1].length();
        chat[i].role    = str[i*2 + 0].c_str();
        chat[i].content = str[i*2 + 1].c_str();
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size * 2);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), true, buf.data(), buf.size());

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), true, buf.data(), buf.size());
    }

    const std::string formatted_chat(buf.data(), res);

    LOG_VERBOSE("formatted_chat", {{"text", formatted_chat.c_str()}});
    printf("formatted_chat: %s\n", formatted_chat.c_str());
    return formatted_chat;
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    std::stringstream chatcmplid;
    chatcmplid << "chatcmpl-" << random_string();

    return chatcmplid.str();
}

//
// other common utils
//

static size_t common_part(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += llama_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;

    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context * ctx, const std::vector<completion_token_output> & probs) {
    json out = json::array();

    for (const auto & prob : probs) {
        json probs_for_token = json::array();

        for (const auto & p : prob.probs) {
            const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }

        const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json {
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }

    return out;
}

//
// OAI utils
//


static std::string rubra_format_python_function_call_str(const std::vector<json> & functions, json & tool_name_map) {
    std::string final_str = "You have access to the following tools:\n";
    printf("rubra_format_function_call_str parsing...\n");
    json type_mapping = {
        {"string", "str"},
        {"integer", "int"},
        {"number", "float"},
        {"float", "float"},
        {"object", "Dict[str, Any]"},
        {"array", "List"},
        {"boolean", "bool"},
        {"null", "None"}
    };
    std::vector<std::string> function_definitions;
    for (const auto & function : functions) {
        const auto &spec = function.contains("function") ? function["function"] : function;
        std::string func_name = spec.value("name", "");
        if (func_name.find('-') != std::string::npos) {
            const std::string origin_func_name = func_name;
            std::replace(func_name.begin(), func_name.end(), '-', '_'); // replace "-" with "_" because - is invalid in python func name
            tool_name_map[func_name] = origin_func_name;
        }
        const std::string description = spec.contains("description") ? spec["description"].get<std::string>() : "";
        const auto& parameters = spec.contains("parameters") && spec["parameters"].contains("properties")? spec["parameters"].value("properties", json({})) : json({});
        const auto& required_params = spec.contains("parameters") && spec["parameters"].contains("properties")? spec["parameters"].value("required", std::vector<std::string>()) : std::vector<std::string>();

        std::vector<std::string> func_args;
        for (auto it = parameters.begin(); it != parameters.end(); ++it) {
            const std::string param = it.key();
            const json& details = it.value();
            std::string json_type = details["type"].get<std::string>();
            std::string python_type = type_mapping.value(json_type, "Any");
            // TODO: handle the case array: should provide more details about type, such as List[str]
            if (details.contains("enum")) {
                python_type = "str";
            }
            std::string arg_str = param + ": " + python_type;
            if (find(required_params.begin(), required_params.end(), param) == required_params.end()) {
                arg_str += " = None";
            }
            func_args.push_back(arg_str);
        }
        std::string func_args_str;
        for (const auto& arg : func_args) {
            if (!func_args_str.empty()) func_args_str += ", ";
            func_args_str += arg;
        }
        
        // Generating Python-like docstring
        std::string docstring = "    \"\"\"\n    " + description + "\n\n";
        for (auto it = parameters.begin(); it != parameters.end(); ++it) {
            const std::string param = it.key();
            const json& details = it.value();
            const std::string& required_text = find(required_params.begin(), required_params.end(), param) != required_params.end() ? "" : "(Optional)";
            std::string param_description = "";
            if (details.count("description") > 0) {
                param_description = details["description"]; // Assuming the description is the first element
            }
            if (details.count("enum") > 0) {
                std::string enum_values;
                for (const std::string val : details["enum"]) {
                    if (!enum_values.empty()) {
                        enum_values += " or ";
                    }
                    enum_values = enum_values+ "\"" + val + "\"";
                }
                if (details["enum"].size() == 1) {
                    param_description += " Only Acceptable value is: " + enum_values;
                } else {
                    param_description += " Only Acceptable values are: " + enum_values;
                }
            }
            if (param_description.empty()) {
                param_description = "No description provided.";
            }
            docstring += "    :param " + param + ": " + param_description + " " + required_text + "\n";
            std::string param_type = details["type"].get<std::string>();
            docstring += "    :type " + param + ": " + type_mapping.value(param_type, "Any") + "\n";
        }
        docstring += "    \"\"\"\n";
        
        // Keeping the function definition in Python format
        std::string function_definition = "def " + func_name + "(" + func_args_str + "):\n" + docstring;
        function_definitions.push_back(function_definition);
    }

    for (const auto& def : function_definitions) {
        final_str += def + "\n";
    }
    final_str += "Use the following format if using tools:\n<<functions>>[toolname1(arg1=value1, arg2=value2, ...), toolname2(arg1=value1, arg2=value2, ...)]";
    return final_str;
}


// Helper function to join strings with a delimiter
static std::string helper_join(const std::vector<std::string>& elements, const std::string& delimiter) {
    std::string result;
    for (auto it = elements.begin(); it != elements.end(); ++it) {
        if (!result.empty()) {
            result += delimiter;
        }
        result += *it;
    }
    return result;
}

static std::string rubra_format_typescript_function_call_str(const std::vector<json> &functions, json &tool_name_map) {
    std::string final_str = "You have access to the following tools:\n";
    json type_mapping = {
        {"string", "string"},
        {"integer", "number"},
        {"number", "number"},
        {"float", "number"},
        {"object", "any"},
        {"array", "any[]"},
        {"boolean", "boolean"},
        {"null", "null"}
    };

    std::vector<std::string> function_definitions;
    for (const auto &function : functions) {
        const auto &spec = function.contains("function") ? function["function"] : function;
        std::string func_name = spec.value("name", "");
        if (func_name.find('-') != std::string::npos) {
            const std::string origin_func_name = func_name;
            std::replace(func_name.begin(), func_name.end(), '-', '_'); // replace "-" with "_" because - is invalid in typescript func name
            tool_name_map[func_name] = origin_func_name;
        }

        const std::string description = spec.contains("description") ? spec["description"].get<std::string>() : "";
        const auto& parameters = spec.contains("parameters") ? spec["parameters"].value("properties", json({})) : json({});
        const auto& required_params = spec.contains("parameters") ? spec["parameters"].value("required", std::vector<std::string>()) : std::vector<std::string>();

        std::vector<std::string> func_args;
        std::string docstring = "/**\n * " + description + "\n";

        for (auto it = parameters.begin(); it != parameters.end(); ++it) {
            const std::string param = it.key();
            const json& details = it.value();
            std::string json_type = details["type"].get<std::string>();
            std::string ts_type = type_mapping.value(json_type, "any");
            std::string param_description = "";
            if (details.count("description") > 0) {
                param_description = details["description"]; // Assuming the description is the first element
            }
            if (details.count("enum") > 0) {
                std::string enum_values;
                for (const std::string val : details["enum"]) {
                    if (!enum_values.empty()) {
                        enum_values += " or ";
                    }
                    enum_values = enum_values+ "\"" + val + "\"";
                }
                if (details["enum"].size() == 1) {
                    param_description += " Only Acceptable value is: " + enum_values;
                } else {
                    param_description += " Only Acceptable values are: " + enum_values;
                }
            }
            if (param_description.empty()) {
                param_description = "No description provided.";
            }
            if (details.contains("enum")) {
                ts_type = "string"; // Enum is treated as string in typescript
            }
            std::string arg_str = param + ": " + ts_type;
            if (find(required_params.begin(), required_params.end(), param) == required_params.end()) {
                arg_str = param + "?: " + ts_type;
                docstring += " * @param " + param + " - " + param_description + "\n";
            } else {
                docstring += " * @param " + param + " - " + param_description + "\n";
            }
            func_args.push_back(arg_str);
        }
        docstring += " */\n";

        std::string func_args_str = helper_join(func_args, ", ");
        std::string function_definition = docstring + "function " + func_name + "(" + func_args_str + "): any {}";

        function_definitions.push_back(function_definition);
    }

    for (const auto& def : function_definitions) {
        final_str += def + "\n\n";
    }
    final_str += "Use the following format if using tools:\n<<functions>>[toolname1(arg1=value1, arg2=value2, ...), toolname2(arg1=value1, arg2=value2, ...)]";
    return final_str;
}



static std::string default_tool_formatter(const std::vector<json>& tools) {
    std::string toolText = "";
    std::vector<std::string> toolNames;
    for (const auto& tool : tools) {
        json function = tool["function"];
        std::string name = function["name"];
        std::string description = function["description"];
        json parameters = function["parameters"]["properties"];

        toolText += "> Tool Name: " + name + "\nTool Description: " + description + "\nTool Args:\n";
        for (auto& [key, value] : parameters.items()) {
            std::string paramType = value["type"];
            std::string paramDesc = value.value("description", "");
            bool required = function["parameters"]["required"].contains(key);
            std::string enumValues = "";

            if (value.contains("enum")) {
                enumValues = ", should be one of [";
                for (const auto& enumValue : value["enum"]) {
                    enumValues += enumValue.get<std::string>() + ", ";
                }
                enumValues.pop_back(); // Remove last comma
                enumValues.pop_back(); // Remove last space
                enumValues += "]";
            }

            toolText += "  - " + key + " (" + paramType + (required ? ", required" : "") + "): " + paramDesc + enumValues + "\n";
        }

        toolNames.push_back(name);
    }

    std::string toolNamesString = "";
    for (const auto& toolName : toolNames) {
        if (!toolNamesString.empty()) {
            toolNamesString += ", ";
        }
        toolNamesString += toolName;
    }

    std::string formattedPrompt = "You have access to the following tools:\n" + toolText +
                                  "Use the following format if using a tool:\n" +
                                  "Action: tool name (one of [" + toolNamesString + "]).\n" +
                                  "Action Input: {'arg1':'value1', 'arg2':'value2', ...}\n";
    return formattedPrompt;
}


static json oaicompat_completion_params_parse(
    const struct llama_model * model,
    const json & body, /* openai api json semantics */
    const std::string & chat_template) {
    json llama_params;

    llama_params["__oaicompat"] = true;

    std::string function_str = "";
    json tool_name_map;

    if (body.contains("tools") && !body["tools"].empty()) {
        // function_str = default_tool_formatter(body["tool"]);
        function_str = rubra_format_typescript_function_call_str(body["tools"], tool_name_map);
    }
    // If 'tool' is not set or empty, check 'functions'
    else if (body.contains("functions") && !body["functions"].empty()) {
    //    function_str = default_tool_formatter(body["functions"]);
       function_str = rubra_format_typescript_function_call_str(body["functions"], tool_name_map);
    }
    printf("\n=============Formatting Input from OPENAI format...============\n");
    if (function_str != "") {
        const std::vector<json> expand_messages = [&]() {
            // std::vector<json> temp_vec = body["messages"];
            // if (body["messages"][0]["role"] == "system") {
            //     std::string old_content = temp_vec[0]["content"];
            //     temp_vec[0]["content"] = old_content + "\n" + function_str;
            // }
            // else {
            //     json function_call;
            //     function_call["role"] = "system";
            //     function_call["content"] = "You are a helpful assistant.\n" + function_str;
            //     temp_vec.push_back(function_call);
            // }
            std::vector<json> temp_vec;
            nlohmann::ordered_map<std::string, std::string> func_observation_map;
            for (size_t i = 0; i < body["messages"].size(); ++i) {

                if (body["messages"][i]["role"] != "tool" and func_observation_map.size() > 0) {
                    // insert the observation from the tool call before the next message
                    std::string observation_str = "";
                    for (const auto& [key, value] : func_observation_map) {
                        if (observation_str != "") {
                            observation_str += ", ";
                        }
                        observation_str += value;
                    }
                    observation_str = std::string("<<observation>>") + "[" + observation_str + "]";
                    json observation_call;
                    observation_call["role"] = "observation";
                    observation_call["content"] = observation_str;
                    temp_vec.push_back(observation_call);
                    func_observation_map.clear();
                }

                if (i == 0){
                    if (body["messages"][0]["role"] == "system") {
                        std::string old_content = body["messages"][0]["content"];
                        json function_call;
                        function_call["role"] = "system";
                        function_call["content"] = old_content + "\n" + function_str;
                        temp_vec.push_back(function_call);
                    }
                    else { // insert a system message of tool definition before the first message
                        json function_call;
                        function_call["role"] = "system";
                        function_call["content"] = "You are a helpful assistant.\n" + function_str;
                        temp_vec.push_back(function_call);
                        temp_vec.push_back(body["messages"][0]);
                    }
                }
                // else if (body["messages"][i]["role"] == "assistant" and (body["messages"][i]["content"].is_null() or body["messages"][i]["content"]=="") and !body["messages"][i]["tool_calls"].is_null() and !body["messages"][i]["tool_calls"].empty()){
                else if (body["messages"][i]["role"] == "assistant" and body["messages"][i].contains("tool_calls")){
                    // convert OpenAI function call format to Rubra format
                    std::string tool_call_str = "";
                    for (const auto & tool_call : body["messages"][i]["tool_calls"]) {
                        std::string func_str = "";
                        func_observation_map[tool_call["id"].get<std::string>()] = ""; // initialize with empty value and later should be updated with the actual value from "tool_call" role message
                        json args = json::parse(tool_call["function"]["arguments"].get<std::string>()); // TODO: catch the exceptions 
                        for (auto& arg : args.items()) {
                            if (func_str != "") {
                                func_str += ", ";
                            }
                            func_str += arg.key() + "=" + arg.value().dump();
                        }
                        func_str = tool_call["function"]["name"].get<std::string>() + "(" + func_str + ")";
                        if (tool_call_str != "") {
                            tool_call_str += ", ";
                        }
                        tool_call_str += func_str;
                    }
                    tool_call_str = std::string("<<functions>>") + "[" + tool_call_str + "]";

                    json function_call;
                    function_call["role"] = "function";
                    function_call["content"] = tool_call_str;
                    temp_vec.push_back(function_call);
                }
                else if (body["messages"][i]["role"] == "tool") {
                    std::string tool_call_id = body["messages"][i]["tool_call_id"].get<std::string>();
                    if (func_observation_map.find(tool_call_id) != func_observation_map.end()) {
                        func_observation_map[tool_call_id] = body["messages"][i]["content"].get<std::string>();
                    } else {
                        LOG_ERROR("Tool call id not found in the map", {{"tool_call_id", tool_call_id}});
                        // TODO: the input is not valid in this case, should return an error
                    }

                }
                else {
                    temp_vec.push_back(body["messages"][i]);
                }
                
            }
            if (func_observation_map.size() > 0) {
                // insert the observation from the tool call before the next message
                std::string observation_str = "";
                for (const auto& [key, value] : func_observation_map) {
                    if (observation_str != "") {
                        observation_str += ", ";
                    }
                    observation_str += value;
                }
                observation_str = std::string("<<observation>>") + "[" + observation_str + "]";
                json observation_call;
                observation_call["role"] = "observation";
                observation_call["content"] = observation_str;
                temp_vec.push_back(observation_call);
                func_observation_map.clear();
            }
            return temp_vec;
        }();
        llama_params["prompt"] = format_chat(model, chat_template, expand_messages);
    }
    else {
        llama_params["prompt"] = format_chat(model, chat_template, body["messages"]);
    }
    llama_params["tool_name_map"] = tool_name_map;

    // Map OpenAI parameters to llama.cpp parameters
    //
    // For parameters that are defined by the OpenAI documentation (e.g.
    // temperature), we explicitly specify OpenAI's intended default; we
    // need to do that because sometimes OpenAI disagrees with llama.cpp
    //
    // https://platform.openai.com/docs/api-reference/chat/create
    llama_sampling_params default_sparams;
    llama_params["model"]             = json_value(body,   "model",             std::string("unknown"));
    llama_params["cache_prompt"]      = json_value(body,   "cache_prompt",      false);
    llama_params["temperature"]       = json_value(body,   "temperature",       0.0);
    llama_params["top_k"]             = json_value(body,   "top_k",             default_sparams.top_k);
    llama_params["top_p"]             = json_value(body,   "top_p",             1.0);
    llama_params["n_predict"]         = json_value(body,   "max_tokens",        -1);
    llama_params["logit_bias"]        = json_value(body,   "logit_bias",        json::object());
    llama_params["frequency_penalty"] = json_value(body,   "frequency_penalty", 0.0);
    llama_params["presence_penalty"]  = json_value(body,   "presence_penalty",  0.0);
    llama_params["seed"]              = json_value(body,   "seed",              LLAMA_DEFAULT_SEED);
    llama_params["stream"]            = json_value(body,   "stream",            false);
    llama_params["mirostat"]          = json_value(body,   "mirostat",          default_sparams.mirostat);
    llama_params["mirostat_tau"]      = json_value(body,   "mirostat_tau",      default_sparams.mirostat_tau);
    llama_params["mirostat_eta"]      = json_value(body,   "mirostat_eta",      default_sparams.mirostat_eta);
    llama_params["penalize_nl"]       = json_value(body,   "penalize_nl",       default_sparams.penalize_nl);
    llama_params["typical_p"]         = json_value(body,   "typical_p",         default_sparams.typical_p);
    llama_params["repeat_last_n"]     = json_value(body,   "repeat_last_n",     default_sparams.penalty_last_n);
    llama_params["ignore_eos"]        = json_value(body,   "ignore_eos",        false);
    llama_params["tfs_z"]             = json_value(body,   "tfs_z",             default_sparams.tfs_z);

    if (body.count("grammar") != 0) {
        llama_params["grammar"] = json_value(body, "grammar", json::object());
    }

    // Handle 'stop' field
    if (body.contains("stop") && body["stop"].is_string()) {
        llama_params["stop"] = json::array({body["stop"].get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Ensure there is ChatML-specific end sequence among stop words
    llama_params["stop"].push_back("<|im_end|>");

    return llama_params;
}


static json format_final_response_oaicompat(const json & request, json result, const std::string & completion_id, bool streaming = false) {
    bool stopped_word        = result.count("stopped_word") != 0;
    bool stopped_eos         = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens    = json_value(result, "tokens_evaluated", 0);
    std::string content      = json_value(result, "content", std::string(""));

    std::vector<json> parsed_content = parsePythonFunctionCalls(content, request["tool_name_map"]);
    
    std::string finish_reason = "length";
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json choices;

    if (streaming) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}}});
    } else {
        if (parsed_content.empty()) {
            choices = json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"message", json{{"content", content},
                                                         {"role", "assistant"}}}}});
        } else {
            std::vector<json> oai_format_tool_calls;
            for (size_t i = 0; i < parsed_content.size(); ++i) {
                const auto &pc = parsed_content[i];
                // Use 'pc' and 'i' as needed
                json tool_call;
                tool_call["id"] = pc["id"];
                tool_call["type"] = "function";
                tool_call["function"] = json{
                    {"name" , pc["name"]},
                    {"arguments" , pc["kwargs"].dump()},
                };
                printf("format_final_response_oaicompat: tool_call: %s\n", tool_call.dump().c_str());
                oai_format_tool_calls.push_back(tool_call);
            }
            choices = json::array({json{{"finish_reason", "tool_calls"},
                                        {"index", 0},
                                        {"message", json{{"tool_calls", oai_format_tool_calls},
                                                         {"role", "assistant"}}}}});
        }
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };
    printf("format_final_response_oaicompat: %s\n", res.dump().c_str());

    if (server_verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(json request ,json result, const std::string & completion_id) {
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({result});
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word   = json_value(result, "stopped_word",  false);
    bool stopped_eos    = json_value(result, "stopped_eos",   false);
    bool stopped_limit  = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content",       std::string(""));

    std::vector<json> parsed_content = parsePythonFunctionCalls(content, request["tool_name_map"]);
    std::time_t t = std::time(0);
    if (!parsed_content.empty()) {
        std::vector<json> res;
        json choices1 = json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}}});
        
        json ret = json{
            {"choices", choices1},
            {"created", t},
            {"id",      completion_id},
            {"model",   modelname},
            {"object",  "chat.completion.chunk"}
        };
        res.push_back(ret);

        for (size_t i = 0; i < parsed_content.size(); ++i) {
                const auto &pc = parsed_content[i];
                // Use 'pc' and 'i' as needed
                json tool_call1;
                tool_call1["id"] = pc["id"];
                tool_call1["type"] = "function";
                tool_call1["index"] = i;
                tool_call1["function"] = json{
                    {"name" , pc["name"]},
                    {"arguments" , ""},
                };
                json ret1 = json{
                    {"choices", json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"tool_calls", std::vector<json>{tool_call1}}}}}})
                                            },
                    {"created", t},
                    {"id",      completion_id},
                    {"model",   modelname},
                    {"object",  "chat.completion.chunk"}
                };
                res.push_back(ret1);
                json tool_call2;
                tool_call2["index"] = i;
                tool_call2["function"] = json{
                    {"name" , ""},
                    {"arguments" , pc["kwargs"].dump()},
                };
                json ret2 = json{
                    {"choices", json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"tool_calls", std::vector<json>{tool_call2}}}}}})
                                            },
                    {"created", t},
                    {"id",      completion_id},
                    {"model",   modelname},
                    {"object",  "chat.completion.chunk"}
                };
                res.push_back(ret2);
            }
        return res;
    }

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    

    json choices;

    if (!finish_reason.empty()) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}}});
    } else {
        if (first) {
            if (content.empty()) {
                choices = json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}}});
            } else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{{"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                return std::vector<json>({initial_ret, second_ret});
            }
        } else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({json::object()});
            }

            choices = json::array({json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            }});
        }
    }

    json ret = json {
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };

    return std::vector<json>({ret});
}

static json format_embeddings_response_oaicompat(const json & request, const json & embeddings) {
    json data = json::array();
    int i = 0;
    for (auto & elem : embeddings) {
        data.push_back(json{
            {"embedding", json_value(elem, "embedding", json::array())},
            {"index",     i++},
            {"object",    "embedding"}
        });
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }},
        {"data", data}
    };

    return res;
}

static json format_tokenizer_response(const std::vector<llama_token> & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
