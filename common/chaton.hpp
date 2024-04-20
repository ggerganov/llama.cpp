#pragma once

#include <vector>
#include <string>

#include "llama.h"
#include "log.h"

// Tag the passed message suitabley as expected by the specified chat handshake template
// and the role. If the specified template is not supported logic will return false.
inline bool llama_chat_apply_template_simple(
            const std::string &tmpl,
            const std::string &role,
            const std::string &content,
            std::string &dst,
            bool add_ass) {
    llama_chat_message msg = { role.c_str(), content.c_str() };
    std::vector<char> buf(content.size() * 2); // This may under allot for small messages and over allot for large messages

    int32_t slen =  llama_chat_apply_template(nullptr, tmpl.c_str(), &msg, 1, add_ass, buf.data(), buf.size());
    if (slen == -1) {
        LOG_TEELN("WARN:%s:Unknown template [%s] requested", __func__, tmpl.c_str());
        dst = "";
        return false;
    }
    if ((size_t) slen > buf.size()) {
        LOGLN("INFO:%s:%s:LengthNeeded:%d:BufSizeWas:%zu", __func__, role.c_str(), slen, buf.size());
        buf.resize(slen);
        slen = llama_chat_apply_template(nullptr, tmpl.c_str(), &msg, 1, add_ass, buf.data(), buf.size());
    }

    const std::string tagged_msg(buf.data(), slen);
    LOGLN("INFO:%s:%s:%s", __func__, role.c_str(), tagged_msg.c_str());
    dst = tagged_msg;
    return true;
}

// return what should be the reverse prompt for the given template id
// ie possible end text tag(s) of specified model type's chat query response
inline std::vector<std::string> llama_chat_reverse_prompt(std::string &template_id) {
    std::vector<std::string> rends;

    if (template_id == "chatml") {
        rends.push_back("<|im_start|>user\n");
    } else if (template_id == "llama2") {
        rends.push_back("</s>");
    } else if (template_id == "llama3") {
        rends.push_back("<|eot_id|>");
    }
    return rends;
}
