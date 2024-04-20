#pragma once

#include <vector>
#include <string>

#include "llama.h"
#include "log.h"

inline std::string llama_chat_apply_template_simple(
            const std::string & tmpl,
            const std::string &role,
            const std::string &content,
            bool add_ass) {
    llama_chat_message msg = { role.c_str(), content.c_str() };
    std::vector<llama_chat_message> msgs{ msg };
    std::vector<char> buf(content.size() * 2);

    int32_t slen =  llama_chat_apply_template(nullptr, tmpl.c_str(), msgs.data(), msgs.size(), add_ass, buf.data(), buf.size());
    if ((size_t) slen > buf.size()) {
        buf.resize(slen);
        slen = llama_chat_apply_template(nullptr, tmpl.c_str(), msgs.data(), msgs.size(), add_ass, buf.data(), buf.size());
    }

    const std::string tagged_msg(buf.data(), slen);
    LOGLN("INFO:%s:%s", __func__, tagged_msg.c_str());
    return tagged_msg;
}

// return what should be the reverse prompt for the given template id
// ie possible end text tag(s) of specified model type's chat query response
std::vector<std::string> llama_chat_reverse_prompt(std::string &template_id) {
    std::vector<std::string> rends;

    if (template_id == "chatml") {
        rends.push_back("<|im_start|>user\n");
    } else if (template_id == "llama3") {
        rends.push_back("<|eot_id|>");
    }
    return rends;
}
