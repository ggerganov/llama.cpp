#pragma once

#include "llama.h"
#include "common.h"

#include "json.hpp"

#include <string>
#include <vector>
#include <sstream>
#include <random>

using json = nlohmann::json;

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
