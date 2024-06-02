#pragma once

/**
 * Allows grouping key-value pairs into groups.
 * by Humans for All
 * 
 * Allows one to maintain a bunch of groups, where each group contains key-value pairs in them.
 * The values could belong to any one of these types ie strings, int, float and bool.
 * 
 * It also tries to provide a crude expanded form of array wrt any of the above supported types.
 * For this one needs to define keys using the pattern TheKeyName-0, TheKeyName-1, ....
 */

#include <map>
#include <string>
#include <vector>
#include <variant>
#include <sstream>


#define GKV_DEBUGLOG_ON

#include "log.h"
#define LINFO_LN LOG_TEELN
#ifdef GKV_DEBUGLOG_ON
#define LDBUG LOG
#define LDBUG_LN LOGLN
#else
#define LDBUG_LN(...)
#define LDBUG(...)
#endif
#define LERRR_LN LOG_TEELN
#define LWARN_LN LOG_TEELN


typedef std::variant<std::string, bool, int32_t, int64_t, double> GroupKVData;
typedef std::vector<std::string> MultiPart;
typedef std::map<std::string, std::map<std::string, GroupKVData>> GroupKVMapMapVariant;

class GroupKV {

private:

    GroupKVMapMapVariant gkv = {};

public:

    GroupKV(GroupKVMapMapVariant defaultMap) : gkv(defaultMap) {}

    static std::string joiner(const MultiPart& parts) {
        std::stringstream joined;
        int iCnt = 0;
        for(auto part: parts) {
            if (iCnt != 0) {
                joined << "-";
            }
            iCnt += 1;
            joined << part;
        }
        return joined.str();
    }

    std::string the_type(const GroupKVData &value) {
        if (std::holds_alternative<std::string>(value)) {
            return "string";
        } else if (std::holds_alternative<bool>(value)) {
            return "bool";
        } else if (std::holds_alternative<int32_t>(value)) {
            return "int32_t";
        } else if (std::holds_alternative<int64_t>(value)) {
            return "int64_t";
        } else if (std::holds_alternative<double>(value)) {
            return "double";
        }
        return "unknown";
    }

    std::string to_str(const GroupKVData &value) {
        auto visitor = [](auto value) -> auto {
            std::stringstream ss;
            ss << value;
            return ss.str();
        };
        return std::visit(visitor, value);
    }

    template<typename TypeWithStrSupp>
    std::string to_str(std::vector<TypeWithStrSupp> values) {
        std::stringstream ss;
        ss << "[ ";
        int cnt = 0;
        for(auto value: values) {
            cnt += 1;
            if (cnt != 1) ss << ", ";
            ss << value;
        }
        ss << " ]";
        return ss.str();
    }

    bool group_exists(const std::string &group) {
        if (gkv.find(group) == gkv.end()) {
            return false;
        }
        return true;
    }

    template<typename SupportedDataType>
    void set_value(const std::string &group, const MultiPart &keyParts, const SupportedDataType &value, const std::string &callerName="") {
        auto key = joiner(keyParts);
        auto &gm = gkv[group];
        gm[key] = value;
        LDBUG_LN("DBUG:GKV:%s_%s:%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), key.c_str(), the_type(value).c_str(), to_str(value).c_str());
    }

    // Dump info about the specified group.
    // If group is empty, then dump info about all groups maintained in this instance.
    std::string dump(const std::string &group, const std::string &msgTag = "") {
        std::stringstream ss;
        for (auto gm: gkv) {
            if (!group.empty() && (gm.first != group)) {
                LDBUG_LN("DBUG:GKV:%s:%s:%s:Skipping...", __func__, msgTag.c_str(), gm.first.c_str());
                continue;
            }
            ss << "\n" << msgTag + ":" << gm.first << ":\n";
            for(auto k: gm.second) {
#ifdef GKV_DEBUGLOG_ON
                ss << msgTag + ":" << "\t" << k.first << ":" + the_type(k.second) + ":" << to_str(k.second) << "\n";
#else
                ss << msgTag + ":" << "\t" << k.first << ":" << to_str(k.second) << "\n";
#endif
            }
        }
        return ss.str();
    }

    // If the specified key is missing, an exception will be thrown.
    template<typename SupportedDataType>
    SupportedDataType get_value(const std::string &group, const MultiPart &keyParts) {
        auto key = joiner(keyParts);
        auto gm = gkv[group];
        if (gm.find(key) == gm.end()) {
            std::stringstream ss;
            ss << "WARN:GKV:" << __func__ << ":" << group << ":Key [" << key << "] not found";
            throw std::range_error(ss.str());
        }
        auto value = gm[key];
        return std::get<SupportedDataType>(value);
    }

    // If the specified key is missing, then the provided default value will be returned.
    template<typename SupportedDataType>
    SupportedDataType get_value(const std::string &group, const MultiPart &keyParts, const SupportedDataType &defaultValue, const std::string &callerName="") {
        try {
            auto value = get_value<SupportedDataType>(group, keyParts);
            LDBUG_LN("DBUG:GKV:%s_%s:%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), to_str(keyParts).c_str(), the_type(value).c_str(), to_str(value).c_str());
            return value;
        } catch (std::exception &e) {
        }
        LDBUG_LN("WARN:GKV:%s_%s:%s:%s:%s:%s[default]", __func__, callerName.c_str(), group.c_str(), to_str(keyParts).c_str(), the_type(defaultValue).c_str(), to_str(defaultValue).c_str());
        return defaultValue;
    }

    template<typename SupportedDataType>
    std::vector<SupportedDataType> get_vector(const std::string &group, const MultiPart &keyParts, const std::vector<SupportedDataType> &defaultValue, const std::string &callerName="") {
        auto key = joiner(keyParts);
        auto gm = gkv[group];
        std::vector<SupportedDataType> array;
        int i = 0;
        while(true) {
            std::stringstream ssArrayKey;
            ssArrayKey << key << "-" << i;
            auto arrayKey = ssArrayKey.str();
            if (gm.find(arrayKey) == gm.end()) {
                break;
            }
            array.push_back(std::get<SupportedDataType>(gm[arrayKey]));
            i += 1;
        }
        if (array.empty()) {
            LDBUG_LN("WARN:GKV:%s_%s:%s:%s:%s[default]", __func__, callerName.c_str(), group.c_str(), key.c_str(), to_str(defaultValue).c_str());
            return defaultValue;
        }
        LDBUG_LN("DBUG:GKV:%s_%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), key.c_str(), to_str(array).c_str());
        return array;
    }

};
