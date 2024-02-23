#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "json.hpp"

using json = nlohmann::json;

/**
 * Integration with functionary model: https://github.com/MeetKai/functionary
 * 
 * A typical flow is:
 * - Step 1: user send request to model
 * - Step 2: model send back a response to user
 * - Step 3: model send back another response to function (optional)
 * - Step 4: function send its returned value to model
 * - Step 5: finally, model send final response back to user
 */

#define FUNCTIONARY_FN_PROMPT "// Supported function definitions that should be called when necessary."
#define FUNCTIONARY_RECIP_ALL "all"
#define FUNCTIONARY_RECIP_NONE "no-tool-call"

namespace llama_functionary {

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

inline std::string str_replace(const std::string & original, const std::string & search, const std::string & replacement) {
    size_t pos = original.find(search);
    if (pos != std::string::npos) {
        std::string result = original;
        result.replace(pos, search.length(), replacement);
        return result;
    }
    return original;
}

inline std::vector<std::string> str_split(std::string str, const std::string & delimiter) {
    size_t pos = 0;
    std::string token;
    std::vector<std::string> output;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        token = str.substr(0, pos);
        output.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    output.push_back(str); // the rest
    return output;
}

typedef struct message {
    std::string from; // can be "system", "user", "assistant" or name of function
    std::string recipient = FUNCTIONARY_RECIP_ALL;
    std::string content;
    bool has_stop = false;
    message() {}
    message(json oai_json) {
        from = json_value(oai_json, "role", std::string(""));
        if (from == "tool") {
            // response from function
            from = json_value(oai_json, "tool_call_id", std::string(""));
        }
        content = json_value(oai_json, "content", std::string(""));
    }
    message(std::string & prompt) {
        std::istringstream iss(prompt);
        std::string line;
        std::stringstream ss;
        int i = 0; // line number
        while (std::getline(iss, line)) {
            if (i == 0) {
                from = str_replace(line, "<|from|>", "");
            } else if (i == 1) {
                recipient = str_replace(line, "<|recipient|>", "");
            } else if (i == 2) {
                ss << str_replace(line, "<|content|>", "");
            } else {
                ss << "\n" << line;
            }
            ++i;
        }
        has_stop = ss.str().find("<|stop|>") != std::string::npos;
        content = str_replace(ss.str(), "<|stop|>", "");
    }
    std::string to_prompt() {
        std::stringstream ss;
        ss << "<|from|>" << from << "\n";
        ss << "<|recipient|>" << recipient << "\n";
        ss << "<|content|>" << content;
        if (has_stop) {
            ss << "<|stop|>";
        }
        ss << "\n";
        return ss.str();
    }
} message;

typedef struct function_param {
    std::string name;
    // type can be "string", "boolean", "number" (typescript types)
    // we do not support array for now
    std::string type;
    std::string desc;
    std::vector<json> allowed_values; // dynamic types
    bool required;
    function_param(std::string param_name, json & oai_json) {
        name = param_name;
        type = json_value(oai_json, "type", std::string());
        desc = json_value(oai_json, "description", std::string());
        if (oai_json.count("enum")) {
            allowed_values = oai_json["enum"];
        }
    }
} function_param;

typedef struct function_def {
    std::string name;
    std::string desc;
    std::vector<function_param> params;
    // parameters.type must always be "object"
    function_def(json & oai_json) {
        std::string type = json_value(oai_json, "type", std::string());
        if (type != "function") {
            throw std::runtime_error("Only tool type \"function\" is supported");
        }
        // function
        json inner_json = json_value(oai_json, "function", json::object());
        name = json_value(inner_json, "name", std::string());
        desc = json_value(inner_json, "description", std::string());
        // function.parameters
        json parameters = json_value(inner_json, "parameters", json::object());
        std::string param_type = json_value(parameters, "type", std::string());
        if (param_type != "object") {
            throw std::runtime_error("Only parameters type \"object\" is supported");
        }
        // function.parameters.properties
        json properties = json_value(parameters, "properties", json::object());
        for (auto& it : properties.items()) {
            std::string curr_prop = it.key();
            json data = json_value(properties, curr_prop, json::object());
            function_param param(curr_prop, data);
            params.push_back(param);
        }
        // TODO: add required !!!!!!!!!!!!!!
    }
} function_def;

// convert OAI type to typescript
inline std::string oai_type_to_ts(std::string & type, std::vector<json> & allowed_values) {
    if (!allowed_values.empty()) {
        std::stringstream ss;
        for (size_t i = 0; i < allowed_values.size(); ++i) {
            ss << allowed_values[i];
            if (i < allowed_values.size() - 1) {
                ss << " | ";
            }
        }
        return ss.str();
    }
    // non-enum types
    if (type == "string" || type == "number" || type == "boolean") {
        return type; // natively supported
    } else if (type == "bool") {
        return "boolean";
    } else if (type == "integer" || type == "float" || type == "double") {
        return "number";
    } else {
        throw std::runtime_error("Unsupported type: " + type);
    }
}

inline std::string serialize_function(function_def & fn) {
    std::stringstream ss;
    if (fn.name.empty()) {
        throw std::runtime_error("Function name is empty");
    }
    if (!fn.desc.empty()) {
        // TODO: what if the desc has multiple lines?
        ss << "// " << fn.desc << "\n";
    }
    ss << "type " << fn.name << " = (_: {\n";
    for (auto & param : fn.params) {
        if (!param.desc.empty()) {
            ss << "// " << param.desc << "\n";
        }
        ss << param.name << ": " << oai_type_to_ts(param.type, param.allowed_values) << ",\n";
    }
    // only support "any" return type for now
    ss << "}) => any;\n\n";
    return ss.str();
}

///////////////////////////////////////////
// Main hooks, to be called in oai.hpp

inline std::string convert_oai_to_prompt(const json & body) {
    std::stringstream ss;
    // convert function definitions
    std::vector<json> tools = json_value(body, "tools", json::array());
    if (!tools.empty()) {
        std::stringstream ss_fn;
        ss_fn << FUNCTIONARY_FN_PROMPT << "\n";
        ss_fn << "namespace functions {" << "\n\n";
        for (auto & tool : tools) {
            function_def fn(tool);
            ss_fn << serialize_function(fn);
        }
        ss_fn << "} // namespace functions";
        // construct the message
        message fn_def_msg;
        fn_def_msg.from = "system";
        fn_def_msg.recipient = FUNCTIONARY_RECIP_ALL;
        fn_def_msg.content = ss_fn.str();
        ss << fn_def_msg.to_prompt();
    }
    // convert history
    std::vector<json> messages = json_value(body, "messages", json::array());
    for (auto & msg_json : messages) {
        if (msg_json.count("tool_calls")) {
            // assistant request to function call, now re-passed to history
            std::vector<json> tool_calls = msg_json["tool_calls"];
            for (auto & tc : tool_calls) {
                message msg;
                msg.from = tc["function"]["name"];
                msg.content = tc["function"]["arguments"];
                ss << msg.to_prompt();
            }
        } else {
            // all other types of message
            message msg(msg_json);
            ss << msg.to_prompt();
        }
    }
    return ss.str();
}

// be careful, the assistant output does not have "<|from|>assistant", you need to add it yourself!
inline json convert_response_to_oai_choices(const std::string & content) {
    std::string text_response;
    json tool_calls = json::array();
    // parse all turns
    std::vector<std::string> turns = str_split(content, "<|from|>");
    for (auto & turn : turns) {
        std::string turn_full = "<|from|>" + turn;
        message msg(turn_full);
        if (msg.from != "assistant") {
            continue; // this case should never happen
        }
        if (msg.recipient != FUNCTIONARY_RECIP_ALL && msg.recipient != FUNCTIONARY_RECIP_NONE) {
            // the assistant decide to call a tool (step 3)
            tool_calls.push_back(json{
                {"id", msg.recipient}, // TODO: maybe generate a random part?
                {"type", "function"},
                {"function", json{
                    {"name", msg.recipient},
                    {"arguments", msg.content},
                }},
            });
        } else {
            // the assistant just want to say something (step 2)
            text_response = msg.content;
        }
    }
    // build final response
    json choices = json::array();
    // TODO: technically, functionary can reponse both text + tool_call in one shot. But for some reasons, the original implementation of OpenAI only return only one, not both.
    if (tool_calls.size() > 0) {
        choices.push_back(json{
            {"index", 0},
            {"finish_reason", "tool_calls"},
            {"message", json{
                {"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", tool_calls},
            }},
        });
    } else {
        choices.push_back(json{
            {"index", 0},
            {"finish_reason", "stop"},
            {"message", json{
                {"role", "assistant"},
                {"content", text_response},
            }},
        });
    }
    return choices;
}

} // namespace llama_functionary
