#pragma once

/***
 * Keep chatting with model and needed role tagging using special tokens simple and flexible, while building on existing interactive flow
 * 1. Use a json file to configure the needed tags for each of the supported chat-handshake-template-standard
 *    a. system-prefix, system-suffix,
 *    b. user-prefix, user-suffix,
 *    c. reverse-prompt
 *    d. global-begin-marker, global-end-marker
 *    e. per-msg-begin-marker, per-msg-end-marker
 *    f. is per-msg-begin-marker used for system+user combo
 * 2. Give the below option to user wrt system prompt, this should give the flexibility to either keep system prompt simple or complex in a flexible yet simple way.
 *    a. the system prompt they specify using -f, is used as is with parse_special when tokenising or
 *    b. whether the system prefix and suffix is added, but without parse_special tokenisation of system-prompt provided by user.
 * 3. chat-apply-template uses the json file, which was loaded, to decide on how to generate the tagged messages for tokenisation
 *    a. input: [ { role: message }, { role: message}, ....]
 *    b. output: [ {flag: data}, { flag: data}, {flag: data}, ....]
 *       * flag is whether to do parse_special for this data, during tokenization or not
 *
 */

#include <string>
#include <fstream>
#include <iostream>
#include <json.hpp>

#include "log.h"

using json = nlohmann::json;

json conMeta;

inline bool chaton_meta_load(std::string &fname) {
    std::ifstream f(fname);
    conMeta = json::parse(f);
    return true;
}

inline bool chaton_meta_ok() {
    if (conMeta == nullptr) {
        return false;
    }
    return true;
}

inline void chaton_meta_dump() {
    if (!chaton_meta_ok()) {
        LOG_TEELN("ERRR:%s:ChatOn Meta: Not loaded yet...", __func__);
        return;
    }
    LOG_TEELN("\n\nINFO:%s:ChatOn Meta\n%s", __func__, conMeta.dump(4).c_str());
}
