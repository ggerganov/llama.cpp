#pragma once

/***
 * Keep chatting with model and needed role tagging using special tokens simple and flexible,
 * while building on existing interactive flow and its in-prefix, in-suffix and antiprompt/reverse-promot
 *
 * 1. Use a json file to configure the needed tags for each of the supported chat-handshake-template-standard
 *    a. system-prefix, system-suffix,
 *    b. user-prefix, user-suffix, assistant-prefix
 *       * these override the in-prefix and in-suffix
 *    c. reverse-prompt
 *    d. global-begin, global-end
 *    d. systemuser-1st-user-has-prefix
 *       * if a combination of 1 system message followed by 1 or more user messages is seen,
 *         then include user prefix only if this flag is set.
 *       * one or two models which I looked at seem to require not just BoS, but also the user-role-tag-prefix
 *         to also be controlled wrt this case. So not differentiating between BoS and any user-role-tag-prefix
 *         However if this needs to be decoupled, then maybe will add begin and end keys to role blocks in the json.
 *         then depending on what model needs, one can setup role-begin and role-prefix suitably.
 * 2. Give the below option to user wrt system prompt, this should give the flexibility to either keep system prompt simple or complex in a flexible yet simple way.
 *    a. the system prompt they specify using -f, is used as is with parse_special when tokenising or
 *    b. whether the system prefix and suffix is added, but without parse_special tokenisation of system-prompt provided by user.
 * 3. chat-apply-template uses the json file, which was loaded, to decide on how to generate the tagged messages for tokenisation
 *    a. input: [ { role, message }, { role, message}, ....]
 *    b. output: [ {flag, data}, { flag, data}, {flag, data}, ....]
 *       * flag is whether to do parse_special for this data, during tokenization or not
 *
 */

#include <string>
#include <fstream>
#include <iostream>
#include <json.hpp>

#include "log.h"


const auto K_SYSTEM = "system";
const auto K_USER = "user";
const auto K_PREFIX = "prefix";
const auto K_SUFFIX = "suffix";
const auto K_BEGIN = "begin";
const auto K_END = "end";
const auto K_GLOBAL = "global";
const auto K_SYSTEMUSER_1ST_USER_HAS_PREFIX = "systemuser-1st-user-has-prefix";


using json = nlohmann::json;

json conMeta;

inline bool chaton_meta_load(std::string &fname) {
    std::ifstream f(fname);
    conMeta = json::parse(f);
    return true;
}

inline void _chaton_meta_dump() {
    LOG_TEELN("\n\nINFO:%s:ChatOn Meta\n%s", __func__, conMeta.dump(4).c_str());
}

inline bool chaton_meta_ok() {
    if (conMeta == nullptr) {
        LOG_TEELN("ERRR:%s:ChatOn Meta: Not loaded yet...", __func__);
        return false;
    }
    _chaton_meta_dump();
    return true;
}


// Return user-prefix + msg + user-suffix
// NOTE: This currently doesnt return about which parts of the tagged message contain tags and which parts the user message
inline std::string chaton_tmpl_apply_single(const std::string &tmpl, const std::string &role, const std::string &content) {
    std::stringstream ss;
    ss << conMeta[tmpl][role][K_PREFIX] << content << conMeta[tmpl][role][K_SUFFIX];
    std::string taggedStr = ss.str();
    LOG_TEELN("DBUG:%s:%s:%s:%s", __func__, tmpl.c_str(), role.c_str(), taggedStr.c_str());
    return taggedStr;
}

// global-begin + [role-prefix + msg + role-suffix] + global-end
// if there is a combination of system-user messages,
//    then 1st user message will have user-prefix only if systemuser-1st-user-has-prefix is true
// NOTE: This currently doesnt return about which parts of the tagged message contain tags and which parts the user message
inline std::string chaton_tmpl_apply(const std::string &tmpl, const std::vector<llama_chat_message> &msgs) {
    std::stringstream ss;
    ss << conMeta[tmpl][K_GLOBAL][K_BEGIN];
    int cntSystem = 0;
    int cntUser = 0;
    int cntOthers = 0;
    for(const auto msg: msgs) {
        auto role = msg.role;
        auto content = msg.content;
        auto prefix = conMeta[tmpl][role][K_PREFIX];
        if (role == K_SYSTEM) {
            cntSystem += 1;
            ss << prefix;
        } else if (role == K_USER) {
            cntUser += 1;
            if ((cntSystem == 1) && (cntUser == 1)) {
                if (conMeta[tmpl][K_SYSTEMUSER_1ST_USER_HAS_PREFIX]) {
                    ss << prefix;
                }
            } else {
                ss << prefix;
            }
        } else {
            cntOthers += 1;
            ss << prefix;
        }
        ss << content << conMeta[tmpl][role][K_SUFFIX];
    }
    ss << conMeta[tmpl][K_GLOBAL][K_END];
    std::string taggedMsgs = ss.str();
    LOG_TEELN("DBUG:%s:%s:%s", __func__, tmpl.c_str(), taggedMsgs.c_str());
    return taggedMsgs;
}

inline std::string chaton_tmpl_role_part(const std::string &tmpl, const std::string &role, const std::string &part) {
    std::string got = conMeta[tmpl][role][part];
    LOG_TEELN("DBUG:%s:%s:%s:%s:%s", __func__, tmpl.c_str(), role.c_str(), part.c_str(), got.c_str());
    return got;
}

inline std::string chaton_tmpl_part(const std::string &tmpl, const std::string &part) {
    std::string got = conMeta[tmpl][part];
    LOG_TEELN("DBUG:%s:%s:%s:%s", __func__, tmpl.c_str(), part.c_str(), got.c_str());
    return got;
}
