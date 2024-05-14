#pragma once

/**
 * Helper to load chaton's configurable template data from json file
 * By Humans for All
 * 
 * Any program which wants to load configurable template data from json file,
 * can include this file to get the needed helpers for same.
*/

#include "chaton.hpp"

#include <json.hpp>
using json = nlohmann::ordered_json;


// Get value corresponding to the specified hierarchy/chain of keys.
// Also throw a more informative exception, if it is not found.
template <typename SupportedType>
inline SupportedType json_get(json &j, const std::vector<std::string_view> &keys, const std::string &msgTag) {
    json curJ = j;
    std::stringstream skey;
    int i = 0;
    for(auto key: keys) {
        if (i != 0) skey << "-";
        i += 1;
        skey << key;
        if (curJ.contains(key)) {
            curJ = curJ[key];
        } else {
            std::stringstream ss;
            ss << "ERRR:ChatON:" << __func__ << ":" << msgTag << ":KeyChain [" << skey.str() << "] is missing";
            throw std::runtime_error(ss.str());
        }
    }
    return curJ;
}

// Update/Extend the configurable template data in specified ChatTemplates instance from the specified json file.
// If nullptr is passed wrt ct, then update/extend the global compiled-in configurable template data.
inline bool chaton_meta_load_json(const std::string &fname, ChatTemplates *ct=nullptr) {
    if (ct == nullptr) {
        ct = &gCT;
    }
    std::ifstream f(fname);
    json conMeta = json::parse(f);
    for(auto it=conMeta.begin(); it != conMeta.end(); ++it) {

        auto group = it.key();
        auto curTmpl = conMeta[group];

        std::string globalBegin = json_get<std::string>(curTmpl, { K_GLOBAL, K_BEGIN }, group);
        ct->set_value<std::string>(group, { K_GLOBAL, K_BEGIN }, globalBegin);
        std::string globalEnd = json_get<std::string>(curTmpl, { K_GLOBAL, K_END }, group);
        ct->set_value<std::string>(group, { K_GLOBAL, K_END }, globalEnd);

        std::string systemBegin = json_get<std::string>(curTmpl, { K_SYSTEM, K_BEGIN }, group);
        ct->set_value<std::string>(group, { K_SYSTEM, K_BEGIN }, systemBegin);
        std::string systemPrefix = json_get<std::string>(curTmpl, { K_SYSTEM, K_PREFIX }, group);
        ct->set_value<std::string>(group, { K_SYSTEM, K_PREFIX }, systemPrefix);
        std::string systemSuffix = json_get<std::string>(curTmpl, { K_SYSTEM, K_SUFFIX }, group);
        ct->set_value<std::string>(group, { K_SYSTEM, K_SUFFIX }, systemSuffix);
        std::string systemEnd = json_get<std::string>(curTmpl, { K_SYSTEM, K_END }, group);
        ct->set_value<std::string>(group, { K_SYSTEM, K_END }, systemEnd);

        std::string userBegin = json_get<std::string>(curTmpl, { K_USER, K_BEGIN }, group);
        ct->set_value<std::string>(group, { K_USER, K_BEGIN }, userBegin);
        std::string userPrefix = json_get<std::string>(curTmpl, { K_USER, K_PREFIX }, group);
        ct->set_value<std::string>(group, { K_USER, K_PREFIX }, userPrefix);
        std::string userSuffix = json_get<std::string>(curTmpl, { K_USER, K_SUFFIX }, group);
        ct->set_value<std::string>(group, { K_USER, K_SUFFIX }, userSuffix);
        std::string userEnd = json_get<std::string>(curTmpl, { K_USER, K_END }, group);
        ct->set_value<std::string>(group, { K_USER, K_END }, userEnd);

        std::string assistantBegin = json_get<std::string>(curTmpl, { K_ASSISTANT, K_BEGIN }, group);
        ct->set_value<std::string>(group, { K_ASSISTANT, K_BEGIN }, assistantBegin);
        std::string assistantPrefix = json_get<std::string>(curTmpl, { K_ASSISTANT, K_PREFIX }, group);
        ct->set_value<std::string>(group, { K_ASSISTANT, K_PREFIX }, assistantPrefix);
        std::string assistantSuffix = json_get<std::string>(curTmpl, { K_ASSISTANT, K_SUFFIX }, group);
        ct->set_value<std::string>(group, { K_ASSISTANT, K_SUFFIX }, assistantSuffix);
        std::string assistantEnd = json_get<std::string>(curTmpl, { K_ASSISTANT, K_END }, group);
        ct->set_value<std::string>(group, { K_ASSISTANT, K_END }, assistantEnd);

        std::string reversePrompt = json_get<std::string>(curTmpl, { K_REVERSE_PROMPT }, group);
        ct->set_value<std::string>(group, { K_REVERSE_PROMPT }, reversePrompt);

        bool systemHasSuffix = json_get<bool>(curTmpl, { K_SYSTEMUSER_SYSTEM_HAS_SUFFIX }, group);
        ct->set_value(group, { K_SYSTEMUSER_SYSTEM_HAS_SUFFIX }, systemHasSuffix);
        bool systemHasEnd = json_get<bool>(curTmpl, { K_SYSTEMUSER_SYSTEM_HAS_END }, group);
        ct->set_value(group, { K_SYSTEMUSER_SYSTEM_HAS_END }, systemHasEnd);

        bool userHasBegin = json_get<bool>(curTmpl, { K_SYSTEMUSER_1ST_USER_HAS_BEGIN }, group);
        ct->set_value(group, { K_SYSTEMUSER_1ST_USER_HAS_BEGIN }, userHasBegin);
        bool userHasPrefix = json_get<bool>(curTmpl, { K_SYSTEMUSER_1ST_USER_HAS_PREFIX }, group);
        ct->set_value(group, { K_SYSTEMUSER_1ST_USER_HAS_PREFIX }, userHasPrefix);

    }
    LDBUG_LN("%s", ct->dump("", "DBUG:ChatONMetaLoad:ChatTemplates").c_str());
    return true;
}
