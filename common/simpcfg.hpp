#pragma once

/**
 * Provides a simple direct 1-level only config file logic
 * by Humans for All
 * 
 * This builds on the GroupKV class.
 * 
 * ## File format
 * 
 * It can consist of multiple config groups.
 * * the group name needs to start at the begining of the line.
 * Each group can inturn contain multiple config fields (key:value pairs) wrt that group.
 * * the group fields need to have 1 or more space at the begining of line.
 * 
 * ## Supported data types
 * 
 * The fields can have values belonging to ane one of the below types
 * * strings - enclosed in double quotes
 *             this is also the fallback catch all type, but dont rely on this behaviour.
 * * int - using decimal number system
 * * float - needs to have a decimal point and or e/E
 *           if decimal point is used, there should be atleast one decimal number on its either side
 * * bool - either true or false
 * 
 * It tries to provide a crude expanded form of array wrt any of the above supported types.
 * For this one needs to define keys using the pattern TheKeyName-0, TheKeyName-1, ....
 * 
 */

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <variant>
#include <sstream>
#include <cuchar>

#include "groupkv.hpp"
#include "datautils_string.hpp"



class SimpCfg : public GroupKV {

private:
    std::regex rInt {R"(^[-+]?\d+$)"};
    std::regex rFloat {R"(^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$)"};

public:

    SimpCfg(GroupKVMapMapVariant defaultMap) : GroupKV(defaultMap) {}


    void set_string(const std::string &group, const MultiPart &keyParts, const std::string &value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_bool(const std::string &group, const MultiPart &keyParts, bool value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_bool(const std::string &group, const MultiPart &keyParts, const std::string &value) {
        std::string sValue = str_tolower(value);
        bool bValue = sValue == "true" ? true : false;
        //LDBUG_LN("DBUG:%s:%s:%s:%d", __func__, value.c_str(), sValue.c_str(), bValue);
        set_bool(group, keyParts, bValue);
    }

    void set_int32(const std::string &group, const MultiPart &keyParts, int32_t value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_int32(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto ivalue = strtol(value.c_str(), nullptr, 0);
        set_int32(group, keyParts, ivalue);
    }

    void set_int64(const std::string &group, const MultiPart &keyParts, int64_t value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_int64(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto ivalue = strtoll(value.c_str(), nullptr, 0);
        set_int64(group, keyParts, ivalue);
    }

    void set_double(const std::string &group, const MultiPart &keyParts, double value) {
        set_value(group, keyParts, value, __func__);
    }

    void set_double(const std::string &group, const MultiPart &keyParts, std::string &value) {
        auto dvalue = strtod(value.c_str(), nullptr);
        set_double(group, keyParts, dvalue);
    }


    std::string get_string(const std::string &group, const MultiPart &keyParts, const std::string &defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    bool get_bool(const std::string &group, const MultiPart &keyParts, bool defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    int32_t get_int32(const std::string &group, const MultiPart &keyParts, int32_t defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    int64_t get_int64(const std::string &group, const MultiPart &keyParts, int64_t defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }

    double get_double(const std::string &group, const MultiPart &keyParts, double defaultValue) {
        return get_value(group, keyParts, defaultValue, __func__);
    }


    static void locale_prepare(std::string &sSavedLocale) {
        sSavedLocale = std::setlocale(LC_ALL, nullptr);
        auto sUpdatedLocale = std::setlocale(LC_ALL, "en_US.UTF-8");
        LDBUG_LN("DBUG:%s:Locale:Prev:%s:Cur:%s", __func__, sSavedLocale.c_str(), sUpdatedLocale);
    }

    static void locale_restore(const std::string &sSavedLocale) {
        auto sCurLocale = std::setlocale(LC_ALL, sSavedLocale.c_str());
        LDBUG_LN("DBUG:%s:Locale:Requested:%s:Got:%s", __func__, sSavedLocale.c_str(), sCurLocale);
    }

    void load(const std::string &fname) {
        std::ifstream f {fname};
        if (!f) {
            LERRR_LN("ERRR:SC:%s:%s:failed to load...", __func__, fname.c_str());
            throw std::runtime_error { "ERRR:SimpCfg:File not found" };
        } else {
            LDBUG_LN("DBUG:SC:%s:%s", __func__, fname.c_str());
        }
        std::string group;
        int iLine = 0;
        while(!f.eof()) {
            iLine += 1;
            std::string curL;
            getline(f, curL);
            if (curL.empty()) {
                continue;
            }
            if (curL[0] == '#') {
                continue;
            }
            bool bGroup = !isspace(curL[0]);
            curL = str_trim(curL);
            if (bGroup) {
                curL = str_trim_single(curL, {"\""});
                group = curL;
                LDBUG_LN("DBUG:SC:%s:group:%s", __func__, group.c_str());
                continue;
            }
            auto dPos = curL.find(':');
            if (dPos == std::string::npos) {
                LERRR_LN("ERRR:SC:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
                throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
            }
            auto dEnd = curL.length() - dPos;
            if ((dPos == 0) || (dEnd < 2)) {
                LERRR_LN("ERRR:SC:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
                throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
            }
            std::string key = curL.substr(0, dPos);
            key = str_trim(key);
            key = str_trim_single(key, {"\""});
            std::string value = curL.substr(dPos+1);
            value = str_trim(value);
            value = str_trim(value, {","});
            std::string vtype = "bool";
            auto valueLower = str_tolower(value);
            if ((valueLower.compare("true") == 0) || (valueLower == "false")) {
                set_bool(group, {key}, value);
            } else if (std::regex_match(value, rInt)) {
                vtype = "int";
                set_int64(group, {key}, value);
            } else if (std::regex_match(value, rFloat)) {
                vtype = "float";
                set_double(group, {key}, value);
            } else {
                vtype = "string";
                if (!value.empty() && (value.front() != '"')) {
                    LWARN_LN("WARN:SC:%s:%d:%s:k:%s:v:%s:is this string?", __func__, iLine, group.c_str(), key.c_str(), value.c_str());
                }
                value = str_trim_single(value, {"\""});
                set_string(group, {key}, value);
            }
            //LDBUG_LN("DBUG:SC:%s:%d:kv:%s:%s:%s:%s", __func__, iLine, group.c_str(), key.c_str(), vtype.c_str(), value.c_str());
        }
    }

};
