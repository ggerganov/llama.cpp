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

#include "log.h"


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

std::string str_trim(std::string &sin) {
    std::string sout = sin;
    sout.erase(sout.find_last);
}

bool sc_load(std::string &fname) {
    std::ifstream f {fname};
    if (!f) {
        LOG_TEELN("ERRR:%s:%s:failed to load...", __func__, fname);
        exit(2);
    }
    while(!f.eof()) {
        std::string curl;
        getline(f, curl);
        if (curl.empty()) {
            continue;
        }

    }
}
