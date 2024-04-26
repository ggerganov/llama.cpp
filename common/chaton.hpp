#pragma once

/***
 *
 * ## Overview
 *
 * Helps chat with a model, by allowing role based special token tagging, based on the specified chat-handshake-template-standard.
 * This is used by main, to build on existing interactive flow and its in-prefix, in-suffix and antiprompt/reverse-promot
 *
 * 1. [ToDo] Use a json file to configure the needed tags for each of the supported chat-handshake-template-standard
 *    * global-> begin & end
 *    * system -> begin, prefix, suffix & end
 *    * user -> begin, prefix, suffix & end; assistant -> begin, prefix, suffix & end
 *      * [main] these override the in-prefix (begin+prefix) and in-suffix
 *    c. reverse-prompt
 *      * [main] this adds to any reverese-prompt specified using cmdline
 *    e. systemuser-sys-has-suffix, systemuser-sys-has-end, systemuser-1st-user-has-begin and systemuser-1st-user-has-prefix
 *       * [chaton-tmpl-apply] if a combination of system and user messages/prompts is passed,
 *         then for system messages suffix and end, as well as
 *         for the 1st user message following the 1st system message,
 *         include system suffix and end and user begin and prefix only if corresponding flags is set.
 *       * begin should normally relate to BoS while prefix should relate to Role Identifier tag.
 *         If there is no need for seperate handling of BoS and RoleIdTag, then one could even
 *         set both BoS and RoleIdTag to one of these entries itself.
 *
 * 2. [main] currently the user specified system prompt (-p + -f) is tagged using system role tags,
 *    and inturn this tagged message is tokenized with parse_special flag.
 *    So any special token related tags in the user specified system prompt will get parsed as special.
 *
 * 3. chaton-tmpl-apply uses the json file, which was loaded, to decide on how to generate the tagged messages for tokenisation.
 *    a. input: [ { role, message }, { role, message}, ....]
 *    b. output: currently a single string is returned which contains the tagged message(s).
 *       [later] if it is needed to differentiate between the special tags added by this from user specified prompts/messages,
 *       then return [ {flag, data}, { flag, data}, {flag, data}, ....],
 *       where the flag specifies whether parse_special should be used or not for the corresponding data, during tokenization.
 *
 * ## Adding support for new model / chat-handshake-template-standard
 *
 * 1. Add suitable entries in json for that model/standard
 * 2. Update the flow in chaton-tmpl-apply, as needed.
 *    Try to update and or reuse the generic flow in chaton-tmpl-apply, as much as possible,
 *    before trying to add a custom logic.
 *    If you update the generic flow, cross check if existing json files will need to be updated or not.
 *
 * ## Notes
 *
 * Currently Main doesnt use chaton-tmpl-apply, but only 
 * * chaton-tmpl-apply-single (for system prompt) and
 * * chaton-tmpl-role-part which maps the user prefix, suffix and reverse-prompt to
 *   in-prefix, in-suffix and antiprompt of main.
 * These always adds any role specific prefix and suffix around the passed message.
 *
 * Sample chaton_meta.json includes template info for
 * * llama2, llama3, gemma, chatml, zephyr, deepseek, monarch
 * * llama2 doesnt apply begin+prefix to 1st user msg following system msg
 * * monarch doesnt apply begin to 1st user msg following system msg
 *
 */

#include <string>
#include <fstream>
#include <iostream>
#include <json.hpp>

#include "log.h"
#include "llama.h"

#define LOGXLN LOG_TEELN

const auto K_SYSTEM = "system";
const auto K_USER = "user";
const auto K_ASSISTANT = "assistant";
const auto K_PREFIX = "prefix";
const auto K_SUFFIX = "suffix";
const auto K_BEGIN = "begin";
const auto K_END = "end";
const auto K_GLOBAL = "global";
const auto K_SYSTEMUSER_SYSTEM_HAS_SUFFIX = "systemuser-system-has-suffix";
const auto K_SYSTEMUSER_SYSTEM_HAS_END = "systemuser-system-has-end";
const auto K_SYSTEMUSER_1ST_USER_HAS_BEGIN = "systemuser-1st-user-has-begin";
const auto K_SYSTEMUSER_1ST_USER_HAS_PREFIX = "systemuser-1st-user-has-prefix";
const auto K_REVERSE_PROMPT = "reverse-prompt";


using json = nlohmann::json;

json conMeta;


/**
 * Helps keep user prompt and chat-hs-template tag parts seperate, but in sequence.
 * Inturn gives the flexibility to tokenize with or without parse_special flag, wrt the different parts of the chat msg(s).
 * One could use the triplet of str, get_types and get_partslens to achieve the above mentioned flexibility.
 */
class ChatParts {

    std::vector<std::string> parts = {};
    std::string types = {""};

public:
    // Identify string with special tokens that need to be processed.
    static const auto S = 's';
    // Identify string which shouldnt have special token processing done.
    static const auto N = 'n';
    // Identify no string condition and or ignore string.
    static const auto X = '?';

    ChatParts() : parts{}, types{""} {}

    char last_type() {
        if (types.length() == 0) {
            return ChatParts::X;
        }
        return types[types.length()-1];
    }

    void add_part(char type, const std::string &part) {
        if (last_type() == type) {
            parts[parts.size()-1] += part;
        } else {
            parts.emplace_back(part);
            types += type;
        }
    }

    std::string str() {
        std::string allin = "";
        for(auto part: parts) {
            allin += part;
        }
        return allin;
    }

    std::string get_types() {
        return types;
    }

    std::vector<int> get_partslens() {
        std::vector<int> lens = {};
        for(auto part: parts) {
            lens.push_back(part.length());
        }
        return lens;
    }

    std::string name() {
        return typeid(*this).name();
    }

    void dump() {
        std::string me = name() + ":" + __func__;
        LOGXLN("INFO:%s:NumTypes:%zu", me.c_str(), types.length());
        LOGXLN("INFO:%s:NumParts:%zu", me.c_str(), parts.size());
        LOGXLN("INFO:%s:StrLength:%zu", me.c_str(), str().length());
        if (parts.size() != types.length()) {
            LOG_TEELN("DBUG:%s:Mismatch between parts and types", me.c_str());
        }
        int i = 0;
        for(auto part: parts) {
            LOGXLN("INFO:%s:%c:%s", me.c_str(), types[i], part.c_str());
            i += 1;
        }
    }

};

inline bool chaton_meta_load(std::string &fname) {
    if (conMeta != nullptr) {
        LOGXLN("WARN:%s:ChatOn Meta: overwriting???", __func__);
    }
    std::ifstream f(fname);
    conMeta = json::parse(f);
    return true;
}

inline bool chaton_tmpl_exists(const std::string &tmpl) {
    if (conMeta == nullptr) {
        LOG_TEELN("ERRR:%s:ChatOnMeta: Not loaded yet...", __func__);
        return false;
    }
    try {
        auto tmplData = conMeta[tmpl];
        return true;
    } catch (json::exception &err) {
        LOG_TEELN("WARN:%s:ChatOnMeta: tmpl[%s] not found...", __func__, tmpl.c_str());
        return false;
    }
}

inline std::string chaton_tmpl_role_kv(const std::string &tmpl, const std::string &role, const std::vector<std::string> &keys) {
    std::string got = "";
    std::string sKeys = "";
    for(auto key: keys) {
        try {
            got += conMeta[tmpl][role][key];
        } catch (json::exception &err) {
        }
        sKeys += "+";
        sKeys += key;
    }
    LOGLN("DBUG:%s:%s:%s:%s:%s", __func__, tmpl.c_str(), role.c_str(), sKeys.c_str(), got.c_str());
    return got;
}

inline std::string chaton_tmpl_kv(const std::string &tmpl, const std::string &key) {
    std::string got = conMeta[tmpl][key];
    LOGLN("DBUG:%s:%s:%s:%s", __func__, tmpl.c_str(), key.c_str(), got.c_str());
    return got;
}

inline bool chaton_tmpl_kv_bool(const std::string &tmpl, const std::string &key) {
    bool got = conMeta[tmpl][key];
    LOGLN("DBUG:%s:%s:%s:%d", __func__, tmpl.c_str(), key.c_str(), got);
    return got;
}


// Return user-prefix + msg + user-suffix
inline bool chaton_tmpl_apply_single_ex(
        const std::string &tmpl,
        const std::string &role,
        const std::string &content,
        std::string &tagged,
        std::string &types,
        std::vector<int> &lens
        ) {
    if (!chaton_tmpl_exists(tmpl)) {
        return false;
    }
    ChatParts cp = {};
    std::stringstream ss;
    std::string beginPrefix = chaton_tmpl_role_kv(tmpl, role, {K_BEGIN, K_PREFIX});
    std::string suffixEnd = chaton_tmpl_role_kv(tmpl, role, {K_SUFFIX, K_END});
    cp.add_part(ChatParts::S, beginPrefix);
    cp.add_part(ChatParts::N, content);
    cp.add_part(ChatParts::S, suffixEnd);
    cp.dump();
    ss << beginPrefix << content << suffixEnd;
    tagged = ss.str();
    std::string cpStr = cp.str();
    if (tagged != cpStr) {
        LOG_TEELN("DBUG:%s:Mismatch between CP[%s] and SS[%s]", __func__, cpStr.c_str(), tagged.c_str());
        exit(2);
    }
    LOGLN("DBUG:%s:%s:%s:%s", __func__, tmpl.c_str(), role.c_str(), tagged.c_str());
    types = cp.get_types();
    lens = cp.get_partslens();
    return true;
}

// Return user-prefix + msg + user-suffix, types string and lens vector wrt the parts that make up the returned string
inline size_t chaton_tmpl_apply_single(
        const std::string &tmpl,
        const std::string &role,
        const std::string &content,
        std::string &tagged
        ) {
    std::string types;
    std::vector<int> lens;
    if (!chaton_tmpl_apply_single_ex(tmpl, role, content, tagged, types, lens)) {
        return -1;
    }
    return tagged.size();
}

/**
 * Apply chat-handshake-template for the specified template standard and role.
 * If the passed char array is smaller that that required for the tagged message,
 * * part of the tagged message which fits within dest buffer is copied
 * * the returned value, indicates the size of the tagged message
 * NOTE:
 * * the passed char array should be able to fit the tagged message+0|null char.
 * * if the return value from this function is larger than or equal to destLength,
 *   then you will have to increase the size of the dest buffer, and call this
 *   function a second time, to ensure that one gets the full tagged message.
 */
inline size_t chat_tmpl_apply_single_capi(
        const char *tmpl,
        const char *role,
        const char *content,
        char *dest,
        const size_t destLength
        ) {
    std::string tagged;
    std::string types;
    std::vector<int> lens;
    auto taggedLength = chaton_tmpl_apply_single(tmpl, role, content, tagged);
    if (taggedLength <= 0) {
        return taggedLength;
    }
    if (dest && (destLength > 0)) {
        strlcpy(dest, tagged.c_str(), destLength);
    }
    return taggedLength;
}

// global-begin + [[role-begin] + [role-prefix] + msg + role-suffix] + global-end
// if there is a combination of system-user messages,
//    then 1st user message will have user-prefix only if systemuser-1st-user-has-prefix is true
// NOTE: returns types and lens to help identify the parts of the tagged msg, which relate to passed and added tags
inline bool chaton_tmpl_apply_ex(
        const std::string &tmpl,
        const std::vector<llama_chat_message> &msgs,
        std::string &tagged,
        std::string &types,
        std::vector<int> &lens,
        bool alertAssistantAtEnd
        ) {
    if (!chaton_tmpl_exists(tmpl)) {
        return false;
    }
    ChatParts cp = {};
    std::stringstream ss;
    std::string globalBegin = chaton_tmpl_role_kv(tmpl, K_GLOBAL, {K_BEGIN});
    ss << globalBegin;
    cp.add_part(ChatParts::S, globalBegin);
    int cntSystem = 0;
    int cntUser = 0;
    int cntOthers = 0;
    for(const auto msg: msgs) {
        auto role = msg.role;
        auto content = msg.content;
        std::string begin = chaton_tmpl_role_kv(tmpl, role, {K_BEGIN});
        auto prefix = chaton_tmpl_role_kv(tmpl, role, {K_PREFIX});
        auto suffix = chaton_tmpl_role_kv(tmpl, role, {K_SUFFIX});
        auto end = chaton_tmpl_role_kv(tmpl, role, {K_END});
        if (role == K_SYSTEM) {
            cntSystem += 1;
            ss << begin << prefix;
            cp.add_part(ChatParts::S, begin);
            cp.add_part(ChatParts::S, prefix);
        } else if (role == K_USER) {
            cntUser += 1;
            if ((cntSystem == 1) && (cntUser == 1)) {
                if (conMeta[tmpl][K_SYSTEMUSER_1ST_USER_HAS_BEGIN]) {
                    ss << begin;
                    cp.add_part(ChatParts::S, begin);
                }
                if (conMeta[tmpl][K_SYSTEMUSER_1ST_USER_HAS_PREFIX]) {
                    ss << prefix;
                    cp.add_part(ChatParts::S, prefix);
                }
            } else {
                ss << begin << prefix;
                cp.add_part(ChatParts::S, begin);
                cp.add_part(ChatParts::S, prefix);
            }
        } else {
            cntOthers += 1;
            ss << begin << prefix;
            cp.add_part(ChatParts::S, begin);
            cp.add_part(ChatParts::S, prefix);
        }
        ss << content;
        cp.add_part(ChatParts::N, content);
        if (role == K_SYSTEM) {
            if (chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_SYSTEM_HAS_SUFFIX)) {
                ss << suffix;
                cp.add_part(ChatParts::S, suffix);
            }
            if (chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_SYSTEM_HAS_END)) {
                ss << end;
                cp.add_part(ChatParts::S, end);
            }
        } else {
            ss << suffix << end;
            cp.add_part(ChatParts::S, suffix);
            cp.add_part(ChatParts::S, end);
        }
    }
    if (alertAssistantAtEnd) {
        auto assistantBeginPrefix = chaton_tmpl_role_kv(tmpl, K_ASSISTANT, {K_BEGIN, K_PREFIX});
        ss << assistantBeginPrefix;
        cp.add_part(ChatParts::S, assistantBeginPrefix);
    }
    auto globalEnd = chaton_tmpl_role_kv(tmpl, K_GLOBAL, {K_END});
    ss << globalEnd;
    cp.add_part(ChatParts::S, globalEnd);
    cp.dump();
    tagged = ss.str();
    std::string cpStr = cp.str();
    if (tagged != cpStr) {
        LOG_TEELN("DBUG:%s:Mismatch between CP[%s] and SS[%s]", __func__, cpStr.c_str(), tagged.c_str());
        exit(2);
    }
    LOGLN("DBUG:%s:%s:%s", __func__, tmpl.c_str(), tagged.c_str());
    LOGLN("DBUG:%s:%s:CntSys[%d]:CntUsr[%d]:CntOthers[%d]", __func__, tmpl.c_str(), cntSystem, cntUser, cntOthers);
    types = cp.get_types();
    lens = cp.get_partslens();
    return true;
}

// global-begin + [[role-begin] + [role-prefix] + msg + role-suffix] + global-end
// if there is a combination of system-user messages,
//    then 1st user message will have user-prefix only if systemuser-1st-user-has-prefix is true
inline int32_t chaton_tmpl_apply(
        const std::string &tmpl,
        const std::vector<llama_chat_message> &msgs,
        bool alertAssistantAtEnd,
        std::string &tagged
        ) {
    std::string types;
    std::vector<int> lens;
    if (!chaton_tmpl_apply_ex(tmpl, msgs, tagged, types, lens, alertAssistantAtEnd)) {
        return -1;
    }
    return tagged.size();
}

inline int32_t chaton_tmpl_apply_capi(
        const char *tmpl,
        const struct llama_chat_message *msgs,
        const size_t numMsgs,
        bool alertAssistantAtEnd,
        char *dest,
        int32_t destLength
        ) {
    if ((tmpl == nullptr) || (dest == nullptr)) {
        return -1;
    }
    std::vector<const llama_chat_message *> vMsgs;
    for(size_t i=0; i<numMsgs; i++) {
        vMsgs.push_back(vMsgs[i]);
    }
    std::string taggedMsgs;
    int32_t taggedLength = chaton_tmpl_apply(tmpl, vMsgs, alertAssistantAtEnd, taggedMsgs);
}


/**
 * if tmpl is
 * * empty string, then dump the full loaded chaton-meta
 * * chaton-template-id, then dump contents related to that specific chat-handshake-template-standard
 */
inline void _chaton_meta_dump(std::string &tmpl) {
    json theJson;
    if (tmpl.empty()) {
        theJson = conMeta;
    } else {
        theJson = conMeta[tmpl];
    }
    LOGXLN("\n\nINFO:%s:ChatOn Meta\n%s", __func__, theJson.dump(4).c_str());
    if (!tmpl.empty()) {
        std::string globalBegin = conMeta[tmpl][K_GLOBAL][K_BEGIN];
        std::string globalEnd = conMeta[tmpl][K_GLOBAL][K_END];
        std::string systemBegin = conMeta[tmpl][K_SYSTEM][K_BEGIN];
        std::string systemPrefix = conMeta[tmpl][K_SYSTEM][K_PREFIX];
        std::string systemSuffix = conMeta[tmpl][K_SYSTEM][K_SUFFIX];
        std::string systemEnd = conMeta[tmpl][K_SYSTEM][K_END];
        std::string userBegin = conMeta[tmpl][K_USER][K_BEGIN];
        std::string userPrefix = conMeta[tmpl][K_USER][K_PREFIX];
        std::string userSuffix = conMeta[tmpl][K_USER][K_SUFFIX];
        std::string userEnd = conMeta[tmpl][K_USER][K_END];
        std::string assistantBegin = conMeta[tmpl][K_ASSISTANT][K_BEGIN];
        std::string assistantPrefix = conMeta[tmpl][K_ASSISTANT][K_PREFIX];
        std::string assistantSuffix = conMeta[tmpl][K_ASSISTANT][K_SUFFIX];
        std::string assistantEnd = conMeta[tmpl][K_ASSISTANT][K_END];

        LOGXLN("INFO:%s:%s:%s", __func__, "global->begin", globalBegin.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "global->end", globalEnd.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "system->begin", systemBegin.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "system->prefix", systemPrefix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "system->suffix", systemSuffix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "system->end", systemEnd.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "user->begin", userBegin.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "user->prefix", userPrefix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "user->suffix", userSuffix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "user->end", userEnd.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "assistant->begin", assistantBegin.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "assistant->prefix", assistantPrefix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "assistant->suffix", assistantSuffix.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, "assistant->end", assistantEnd.c_str());
        LOGXLN("INFO:%s:%s:%s", __func__, K_REVERSE_PROMPT, chaton_tmpl_kv(tmpl, K_REVERSE_PROMPT).c_str());
        LOGXLN("INFO:%s:%s:%d", __func__, K_SYSTEMUSER_SYSTEM_HAS_SUFFIX, chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_SYSTEM_HAS_SUFFIX));
        LOGXLN("INFO:%s:%s:%d", __func__, K_SYSTEMUSER_SYSTEM_HAS_END, chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_SYSTEM_HAS_END));
        LOGXLN("INFO:%s:%s:%d", __func__, K_SYSTEMUSER_1ST_USER_HAS_BEGIN, chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_1ST_USER_HAS_BEGIN));
        LOGXLN("INFO:%s:%s:%d", __func__, K_SYSTEMUSER_1ST_USER_HAS_PREFIX, chaton_tmpl_kv_bool(tmpl, K_SYSTEMUSER_1ST_USER_HAS_PREFIX));

        if (!userEnd.empty()) {
            LOG_TEELN("WARN:%s:User->End seems to be set to [%s], do cross check if this is proper and needed", __func__, userEnd.c_str());
        }
        if (!assistantBegin.empty()) {
            LOG_TEELN("WARN:%s:Assistant->Begin seems to be set to [%s], do cross check if this is proper and needed", __func__, assistantBegin.c_str());
        }
    }
}

/**
 * Check that a meta-json file has been loaded.
 * Verify that specified chaton-template-id contains required fields in meta-json, using meta-dump
 */
inline bool chaton_meta_ok(std::string &tmpl) {
    if (conMeta == nullptr) {
        LOG_TEELN("ERRR:%s:ChatOn Meta: Not loaded yet...", __func__);
        return false;
    }
    _chaton_meta_dump(tmpl);
    return true;
}
