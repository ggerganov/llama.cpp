#pragma once

/**
 * Provide a simple config file logic
 * It can consist of multiple config groups.
 * * the group name needs to start at the begining of the line.
 * Each group can inturn contain multiple config fields wrt that group.
 * * the group fields need to have 1 or more space at the begining of line.
 * 
 */

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <variant>
#include <sstream>


#define SC_DEBUG
#define SC_TEST_PRG
#ifdef SC_TEST_PRG
#define LDBUG_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#define LERRR_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#define LWARN_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#else
#include "log.h"
#define LDBUG_LN LOGLN
#define LERRR_LN LOG_TEELN
#define LWARN_LN LOG_TEELN
#endif


std::string str_trim(std::string sin, std::string trimChars=" \t\n") {
    sin.erase(sin.find_last_not_of(trimChars)+1);
    sin.erase(0, sin.find_first_not_of(trimChars));
    return sin;
}

// Remove atmost 1 char at the begin and 1 char at the end of the passed string,
// provided the char belongs to one of the chars in trimChars.
// NOTE: Not sure this handles non english utf8 multibyte chars properly,
// need to cross check.
std::string str_trim_single(std::string sin, std::string trimChars=" \t\n") {
    if (sin.empty()) return sin;
    for(auto c: trimChars) {
        if (c == sin.front()) {
            sin = sin.substr(1, std::string::npos);
            break;
        }
    }
    if (sin.empty()) return sin;
    for(auto c: trimChars) {
        if (c == sin.back()) {
            sin = sin.substr(0, sin.length()-1);
            break;
        }
    }
    return sin;
}

std::string str_tolower(const std::string &sin) {
    std::string sout;
    std::transform(sin.begin(), sin.end(),sout.begin(), [](char c)->char {return std::tolower(c);});
    return sout;
}


typedef std::variant<std::string, bool, int64_t, double> SimpCfgData;

class SimpCfg {

private:
    std::map<std::string, std::map<std::string, SimpCfgData>> mapV = {};
    std::regex rInt {R"(^[-+]?\d+$)"};
    std::regex rFloat {R"(^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$)"};

public:

    std::string to_str(const SimpCfgData &value) {
        auto visitor = [](auto value) -> auto {
            std::stringstream ss;
            ss << value;
            return ss.str();
        };
        return std::visit(visitor, value);
    }

    template<typename SupportedDataType>
    void set_value(const std::string &group, const std::string &key, const SupportedDataType &value, const std::string &callerName="") {
        auto &gm = mapV[group];
        gm[key] = value;
        std::stringstream ss;
        ss << value;
        LDBUG_LN("DBUG:SC:%s_%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), key.c_str(), ss.str().c_str());
    }

    void set_string(const std::string &group, const std::string &key, const std::string &value) {
        set_value(group, key, value, __func__);
    }

    void set_bool(const std::string &group, const std::string &key, bool value) {
        set_value(group, key, value, __func__);
    }

    void set_bool(const std::string &group, const std::string &key, const std::string &value) {
        set_bool(group, key, str_tolower(value) == "true" ? true : false);
    }

    void set_int64(const std::string &group, const std::string &key, int64_t value) {
        set_value(group, key, value, __func__);
    }

    void set_int64(const std::string &group, const std::string &key, std::string &value) {
        auto ivalue = strtoll(value.c_str(), nullptr, 0);
        set_int64(group, key, ivalue);
    }

    void set_double(const std::string &group, const std::string &key, double value) {
        set_value(group, key, value, __func__);
    }

    void set_double(const std::string &group, const std::string &key, std::string &value) {
        auto dvalue = strtod(value.c_str(), nullptr);
        set_double(group, key, dvalue);
    }

    template<typename SupportedDataType>
    SupportedDataType get_value(const std::string &group, const std::string &key, const SupportedDataType &defaultValue, const std::string &callerName="") {
        auto gm = mapV[group];
#ifdef SC_DEBUG
        for(auto k: gm) {
            LDBUG_LN("DBUG:SC:%s:Iterate:%s:%s", __func__, k.first.c_str(), to_str(k.second).c_str());
        }
#endif
        if (gm.find(key) == gm.end()) {
            std::stringstream ss;
            ss << defaultValue;
            LWARN_LN("DBUG:SC:%s_%s:%s:%s:%s[default]", __func__, callerName.c_str(), group.c_str(), key.c_str(), ss.str().c_str());
            return defaultValue;
        }
        auto value = gm[key];
        LDBUG_LN("DBUG:SC:%s:%s:%s:%s", __func__, group.c_str(), key.c_str(), to_str(value).c_str());
        return std::get<SupportedDataType>(value);
    }

    std::string get_string(const std::string &group, const std::string &key, const std::string &defaultValue) {
        return get_value(group, key, defaultValue, __func__);
    }

    bool get_bool(const std::string &group, const std::string &key, bool defaultValue) {
        return get_value(group, key, defaultValue, __func__);
    }

    int64_t get_int64(const std::string &group, const std::string &key, int64_t defaultValue) {
        return get_value(group, key, defaultValue, __func__);
    }

    double get_double(const std::string &group, const std::string &key, double defaultValue) {
        return get_value(group, key, defaultValue, __func__);
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
                curL = str_trim_single(curL, "\"");
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
            key = str_trim_single(key, "\"");
            std::string value = curL.substr(dPos+1);
            value = str_trim(value);
            value = str_trim(value, ",");
            std::string vtype = "bool";
            if ((value == "true") || (value == "false")) {
                set_bool(group, key, value);
            } else if (std::regex_match(value, rInt)) {
                vtype = "int";
                set_int64(group, key, value);
            } else if (std::regex_match(value, rFloat)) {
                vtype = "float";
                set_double(group, key, value);
            } else {
                vtype = "string";
                if (!value.empty() && (value.front() != '"')) {
                    LWARN_LN("WARN:SC:%s:kv:is this string:%s", __func__, value.c_str());
                }
                value = str_trim_single(value, "\"");
                set_string(group, key, value);
            }
            //LDBUG_LN("DBUG:SC:%s:kv:%s:%s:%s:%s", __func__, group.c_str(), key.c_str(), vtype.c_str(), value.c_str());
        }
    }

};


#ifdef SC_TEST_PRG
int main(int argc, char **argv) {
    std::string fname {argv[1]};
    SimpCfg sc;
    sc.load(fname);

    sc.get_bool("testme", "key101b", false);
    sc.get_string("testme", "key101s", "Not found");
    sc.get_int64("testme", "key101i", 123456);
    sc.get_double("testme", "key101d", 123456.789);

    sc.set_bool("testme", "key201b", true);
    sc.set_string("testme", "key201s", "hello world");
    sc.set_int64("testme", "key201i", 987654);
    sc.set_double("testme", "key201d", 9988.7766);

    sc.get_bool("testme", "key201b", false);
    sc.get_string("testme", "key201s", "Not found");
    sc.get_int64("testme", "key201i", 123456);
    sc.get_double("testme", "key201d", 123456.789);

    sc.get_string("mistral", "system-prefix", "Not found");
    sc.get_string("\"mistral\"", "\"system-prefix\"", "Not found");
    return 0;
}
#endif
