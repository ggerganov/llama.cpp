#pragma once

/**
 * Provides a simple direct 1-level only config file logic
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
 */

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <regex>
#include <variant>
#include <sstream>
#include <format>
#include <cuchar>


#define SC_DEBUG
#define SC_TEST_PRG
#ifdef SC_TEST_PRG
#define LINFO_LN(FMT, ...) fprintf(stdout, FMT"\n", __VA_ARGS__)
#define LDBUG_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#define LERRR_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#define LWARN_LN(FMT, ...) fprintf(stderr, FMT"\n", __VA_ARGS__)
#else
#include "log.h"
#define LINFO_LN LOG_TEELN
#define LDBUG_LN LOGLN
#define LERRR_LN LOG_TEELN
#define LWARN_LN LOG_TEELN
#endif

#undef SC_STR_OVERSMART
#ifdef SC_STR_OVERSMART
#define str_trim str_trim_oversmart
#else
#define str_trim str_trim_dumb
#endif


size_t wcs_to_mbs(std::string &sDest, const std::wstring &wSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const wchar_t *wSrcP = wSrc.c_str();
    auto reqLen = std::wcsrtombs(nullptr, &wSrcP, 0, &mbState);
    sDest.resize(reqLen);
    return std::wcsrtombs(sDest.data(), &wSrcP, sDest.length(), &mbState);
}

size_t mbs_to_wcs(std::wstring &wDest, std::string &sSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const char *sSrcP = sSrc.c_str();
    auto reqLen = std::mbsrtowcs(nullptr, &sSrcP, 0, &mbState);
    wDest.resize(reqLen);
    return std::mbsrtowcs(wDest.data(), &sSrcP, wDest.length(), &mbState);
}

// Remove chars from begin and end of the passed string, provided the char belongs
// to one of the chars in trimChars.
// NOTE: Chars being trimmed (ie trimChars) needs to be 1byte encoded chars.
// NOTE: This will work provided the string being trimmed as well the chars being
// trimmed are made up of 1byte encoded chars including in utf8 encoding space.
// If the string being trimmed includes multibyte encoded characters at the end,
// then trimming can mess things up.
std::string str_trim_dumb(std::string sin, std::string trimChars=" \t\n") {
    sin.erase(sin.find_last_not_of(trimChars)+1);
    sin.erase(0, sin.find_first_not_of(trimChars));
    return sin;
}

// Remove chars from begin and end of the passed string, provided the char belongs
// to one of the chars in trimChars.
// NOTE: Internally converts to wchar/wstring to try and support proper trimming,
// wrt possibly more languages, to some extent, ie even if the passed string
// contains multibyte encoded characters in it.
std::string str_trim_oversmart(std::string sIn, std::string trimChars=" \t\n") {
    std::wstring wIn;
    mbs_to_wcs(wIn, sIn);
    std::wstring wTrimChars;
    mbs_to_wcs(wTrimChars, trimChars);
    wIn.erase(wIn.find_last_not_of(wTrimChars)+1);
    wIn.erase(0, wIn.find_first_not_of(wTrimChars));
    std::string sOut;
    wcs_to_mbs(sOut, wIn);
    return sOut;
}

// Remove atmost 1 char at the begin and 1 char at the end of the passed string,
// provided the char belongs to one of the chars in trimChars.
// NOTE: Chars being trimmed (ie trimChars) needs to be 1byte encoded chars.
// NOTE: This will work provided the string being trimmed as well the chars being
// trimmed are made up of 1byte encoded chars including in utf8 encoding space.
// If the string being trimmed includes multibyte encoded characters at the end,
// then trimming can mess things up.
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

// This works for 1byte encoded chars, including in utf8 encoding space.
// This wont work for multibyte encoded chars.
std::string str_tolower(const std::string &sin) {
    std::string sout;
    sout.resize(sin.size());
    std::transform(sin.begin(), sin.end(), sout.begin(), [](char c)->char {return std::tolower(c);});
    //LDBUG_LN("DBUG:%s:%s:%s", __func__, sin.c_str(), sout.c_str());
    return sout;
}

void str_compare_dump(const std::string &s1, const std::string &s2) {
    LDBUG_LN("DBUG:%s:%s:Len:%zu", __func__, s1.c_str(), s1.length());
    LDBUG_LN("DBUG:%s:%s:Len:%zu", __func__, s2.c_str(), s2.length());
    int minLen = s1.length() < s2.length() ? s1.length() : s2.length();
    for(int i=0; i<minLen; i++) {
        LDBUG_LN("DBUG:%s:%d:%c:%c", __func__, i, s1[i], s2[i]);
    }
}


template<typename TypeWithStrSupp>
std::string str(TypeWithStrSupp value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template<typename TypeWithStrSupp>
std::string str(std::vector<TypeWithStrSupp> values) {
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
        std::string sValue = str_tolower(value);
        bool bValue = sValue == "true" ? true : false;
        //LDBUG_LN("DBUG:%s:%s:%s:%d", __func__, value.c_str(), sValue.c_str(), bValue);
        set_bool(group, key, bValue);
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

    void dump(const std::string &group) {
        for (auto gm: mapV) {
            if (!group.empty() && (gm.first != group)) {
                LINFO_LN("INFO:SC:%s:%s:Skipping...", __func__, gm.first.c_str());
                continue;
            }
            for(auto k: gm.second) {
                LINFO_LN("DBUG:SC:%s:%s:Iterate:%s:%s", __func__, gm.first.c_str(), k.first.c_str(), to_str(k.second).c_str());
            }
        }
    }

    template<typename SupportedDataType>
    SupportedDataType get_value(const std::string &group, const std::string &key, const SupportedDataType &defaultValue, const std::string &callerName="") {
        auto gm = mapV[group];
        if (gm.find(key) == gm.end()) {
            std::stringstream ss;
            ss << defaultValue;
            LWARN_LN("DBUG:SC:%s_%s:%s:%s:%s[default]", __func__, callerName.c_str(), group.c_str(), key.c_str(), ss.str().c_str());
            return defaultValue;
        }
        auto value = gm[key];
        LDBUG_LN("DBUG:SC:%s_%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), key.c_str(), to_str(value).c_str());
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


    template<typename SupportedDataType>
    std::vector<SupportedDataType> get_vector(const std::string &group, const std::string &key, const std::vector<SupportedDataType> &defaultValue, const std::string &callerName="") {
        auto gm = mapV[group];
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
            LWARN_LN("DBUG:SC:%s_%s:%s:%s:%s[default]", __func__, callerName.c_str(), group.c_str(), key.c_str(), str(defaultValue).c_str());
            return defaultValue;
        }
        LDBUG_LN("DBUG:SC:%s_%s:%s:%s:%s", __func__, callerName.c_str(), group.c_str(), key.c_str(), str(array).c_str());
        return array;
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
            auto valueLower = str_tolower(value);
            if ((valueLower.compare("true") == 0) || (valueLower == "false")) {
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
                    LWARN_LN("WARN:SC:%s:%d:%s:k:%s:v:%s:is this string?", __func__, iLine, group.c_str(), key.c_str(), value.c_str());
                }
                value = str_trim_single(value, "\"");
                set_string(group, key, value);
            }
            //LDBUG_LN("DBUG:SC:%s:%d:kv:%s:%s:%s:%s", __func__, iLine, group.c_str(), key.c_str(), vtype.c_str(), value.c_str());
        }
    }

};


#ifdef SC_TEST_PRG

#include <iostream>

void check_string() {
    std::vector<std::string> vStandard = { "123", "1अ3" };
    std::cout << "**** string **** " << vStandard.size() << std::endl;
    for(auto sCur: vStandard) {
        std::cout << std::format("string: [{}] len[{}] size[{}]", sCur, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::cout << std::format("string:{}:pos:{}:char:{}[0x{:x}]\n", sCur, i, c, (uint8_t)c);
            i += 1;
        }
    }
}

void check_u8string() {
    std::vector<std::u8string> vU8s = { u8"123", u8"1अ3" };
    std::cout << "**** u8string **** " << vU8s.size() << std::endl;
    for(auto sCur: vU8s) {
        std::string sCurx (sCur.begin(), sCur.end());
        std::cout << std::format("u8string: [{}] len[{}] size[{}]", sCurx, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            //std::cout << c << std::endl;
            std::cout << std::format("u8string:{}:pos:{}:char:{}[0x{:x}]\n", sCurx, i, (unsigned char)c, (unsigned char)c);
            i += 1;
        }
    }
}

void check_wstring_wcout() {
    std::wcout.imbue(std::locale("en_US.UTF-8"));
    std::vector<std::wstring> vWide = { L"123", L"1अ3" };
    std::cout << "**** wstring wcout **** " << vWide.size() << std::endl;
    for(auto sCur: vWide) {
        std::wcout << sCur << std::endl;
        std::wcout << std::format(L"wstring: [{}] len[{}] size[{}]", sCur, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::wcout << std::format(L"wstring:{}:pos:{}:char:{}[0x{:x}]\n", sCur, i, c, c);
            i += 1;
        }
    }
}

void check_wstring_cout() {
    std::vector<std::wstring> vWide = { L"123", L"1अ3" };
    std::cout << "**** wstring cout **** " << vWide.size() << std::endl;
    for(auto sCur: vWide) {
        std::string sCury;
        wcs_to_mbs(sCury, sCur);
        std::cout << std::format("wstring: [{}] len[{}] size[{}]", sCury, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::wstringstream wsc;
            wsc << c;
            std::string ssc;
            wcs_to_mbs(ssc, wsc.str());
            std::cout << std::format("wstring:{}:pos:{}:char:{}[0x{:x}]\n", sCury, i, ssc, (uint32_t)c);
            i += 1;
        }
    }
}

void check_strings() {
    std::string sSavedLocale;
    SimpCfg::locale_prepare(sSavedLocale);
    check_string();
    check_u8string();
    //check_wstring_wcout();
    check_wstring_cout();
    SimpCfg::locale_restore(sSavedLocale);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        LERRR_LN("USAGE:%s simp.cfg", argv[0]);
        exit(1);
    }

    check_strings();

    std::string fname {argv[1]};
    SimpCfg sc;
    sc.load(fname);
    sc.dump("");

    sc.get_bool("testme", "key101b", false);
    sc.get_string("testme", "key101s", "Not found");
    sc.get_int64("testme", "key101i", 123456);
    sc.get_double("testme", "key101d", 123456.789);

    sc.set_bool("testme", "key201b", true);
    sc.set_string("testme", "key201s", "hello world");
    sc.set_int64("testme", "key201i", 987654);
    sc.set_double("testme", "key201d", 9988.7766);

    sc.dump("testme");
    sc.get_bool("testme", "key201b", false);
    sc.get_string("testme", "key201s", "Not found");
    sc.get_int64("testme", "key201i", 123456);
    sc.get_double("testme", "key201d", 123456.789);

    sc.get_string("mistral", "system-prefix", "Not found");
    sc.get_string("\"mistral\"", "\"system-prefix\"", "Not found");

    sc.get_vector<int64_t>("testme", "keyA100", {1, 2, 3});
    sc.get_vector<std::string>("testme", "keyA100", { "A", "അ", "अ", "ಅ" });
    sc.set_int64("testme", "keyA300-0", 330);
    sc.set_int64("testme", "keyA300-1", 331);
    sc.set_int64("testme", "keyA300-2", 332);
    sc.set_string("testme", "keyA301-0", "India");
    sc.set_value<std::string>("testme", "keyA301-1", "World");
    sc.set_string("testme", "keyA301-2", "AkashaGanga");
    sc.get_vector<int64_t>("testme", "keyA300", {1, 2, 3});
    sc.get_vector<std::string>("testme", "keyA301", { "yes 1", "No 2", "very well 3" });
    return 0;
}
#endif
