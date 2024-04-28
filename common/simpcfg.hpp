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

#define TEST_LOGIC
#ifdef TEST_LOGIC
#define LOG_TEELN(FMT, ...) printf(FMT"\n", __VA_ARGS__)
#else
#include "log.h"
#endif

std::map<std::string, std::map<std::string, std::string>> mapStrings = {};
std::map<std::string, std::map<std::string, bool>> mapBools = {};

void sc_set_string(std::string &group, std::string &key, std::string &value) {
    auto gm = mapStrings[group];
    gm[key] = value;
}

void sc_set_bool(std::string &group, std::string &key, bool value) {
    auto gm = mapBools[group];
    gm[key] = value;
}

std::string sc_get_string(std::string &group, std::string &key) {
    auto gm = mapStrings[group];
    return gm[key];
}

bool sc_get_bool(std::string &group, std::string &key) {
    auto gm = mapBools[group];
    return gm[key];
}

std::string str_trim(std::string sin, std::string trimChars=" \t\n") {
    sin.erase(sin.find_last_not_of(trimChars)+1);
    sin.erase(0, sin.find_first_not_of(trimChars));
    return sin;
}

void sc_load(std::string &fname) {
    std::ifstream f {fname};
    if (!f) {
        LOG_TEELN("ERRR:%s:%s:failed to load...", __func__, fname.c_str());
        throw std::runtime_error { "ERRR:SimpCfg:File not found" };
    } else {
        LOG_TEELN("DBUG:%s:%s", __func__, fname.c_str());
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
            group = curL;
            LOG_TEELN("DBUG:%s:group:%s", __func__, group.c_str());
            continue;
        }
        auto dPos = curL.find(':');
        if (dPos == std::string::npos) {
            LOG_TEELN("ERRR:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
            throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
        }
        auto dEnd = curL.length() - dPos;
        if ((dPos == 0) || (dEnd < 2)) {
            LOG_TEELN("ERRR:%s:%d:invalid key value line:%s", __func__, iLine, curL.c_str());
            throw std::runtime_error { "ERRR:SimpCfg:Invalid key value line" };
        }
        std::string key = curL.substr(0, dPos);
        key = str_trim(key);
        std::string value = curL.substr(dPos+1);
        value = str_trim(value);
        value = str_trim(value, ",");
        std::string vtype = "bool";
        if ((value == "true") || (value == "false")) {
            sc_set_bool(group, key, value == "true" ? true : false);
        } else {
            vtype = "string";
            sc_set_string(group, key, value);
        }
        LOG_TEELN("DBUG:%s:kv:%s:%s:%s:%s", __func__, group.c_str(), key.c_str(), vtype.c_str(), value.c_str());
    }
}


#ifdef TEST_LOGIC
int main(int argc, char **argv) {
    std::string fname {argv[1]};
    sc_load(fname);
    return 0;
}
#endif
