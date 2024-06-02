#pragma once

/**
 * A bunch of helper routines to work with strings.
 * by Humans for All
 * 
 * ## Some notes for later
 * 
 * NativeCharSize encoded char refers to chars which fit within the size of char type in a given
 * type of c++ string or base bitsize of a encoding standard, like 1 byte in case of std::string,
 * utf-8, ...
 * * example english alphabets in utf-8 encoding space are 1byte chars, in its variable length
 *   encoding space.
 * 
 * MultiNativeCharSize encoded char refers to chars which occupy multiple base-char-bit-size of
 * a c++ string type or char encoding standard.
 * * example indian scripts alphabets in utf-8 encoding space occupy multiple bytes in its variable
 *   length encoding space.
 * 
 * Sane variable length encoding - refers to encoding where the values of NativeCharSized chars of
 * a char encoding space cant overlap with values in NativeCharSize subparts of MultiNativeCharSized
 * chars of the same char encoding standard.
 * * utf-8 shows this behaviour
 * * chances are utf-16 and utf-32 also show this behaviour (need to cross check once)
 * 
*/

#include <string>
#include <iomanip>

#include "log.h"


#undef DUS_DEBUG_VERBOSE

#undef DUS_STR_OVERSMART
#ifdef DUS_STR_OVERSMART
#define str_trim str_trim_oversmart
#else
#define str_trim str_trim_dumb
#endif


inline size_t wcs_to_mbs(std::string &sDest, const std::wstring &wSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const wchar_t *wSrcP = wSrc.c_str();
    auto reqLen = std::wcsrtombs(nullptr, &wSrcP, 0, &mbState);
    if (reqLen == static_cast<std::size_t>(-1)) {
        throw std::runtime_error("ERRR:WCS2MBS:Failed probing of size...");
    }
    sDest.resize(reqLen);
    return std::wcsrtombs(sDest.data(), &wSrcP, sDest.length(), &mbState);
}

inline size_t mbs_to_wcs(std::wstring &wDest, const std::string &sSrc) {
    std::mbstate_t mbState = std::mbstate_t();
    const char *sSrcP = sSrc.c_str();
    auto reqLen = std::mbsrtowcs(nullptr, &sSrcP, 0, &mbState);
    if (reqLen == static_cast<std::size_t>(-1)) {
        throw std::runtime_error("ERRR:MBS2WCS:Failed probing of size...");
    }
    wDest.resize(reqLen);
    return std::mbsrtowcs(wDest.data(), &sSrcP, wDest.length(), &mbState);
}

inline std::string uint8_as_hex(uint8_t c) {
    char hex[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
    std::string out = "00";
    out[0] = hex[((c & 0xf0) >> 4)];
    out[1] = hex[(c & 0x0f)];
    return out;
}

template <typename TString>
inline std::string string_as_hex(const TString &sIn){
    std::stringstream ssout;
    ssout << "[ ";
    for(auto c: sIn) {
        auto cSize = sizeof(c);
        if (cSize == 1) {
            ssout << uint8_as_hex(c) << ", ";
        } else if (cSize == 2) {
            ssout << std::setfill('0') << std::setw(cSize*2) << std::hex << static_cast<uint16_t>(c) << ", ";
        } else if (cSize == 4) {
            ssout << std::setfill('0') << std::setw(cSize*2) << std::hex << static_cast<uint32_t>(c) << ", ";
        } else {
            std::stringstream ss;
            ss << "ERRR:" << __func__ << ":Unsupported char type with size [" << cSize << "]";
            throw std::runtime_error( ss.str().c_str() );
        }
    }
    ssout << " ]";
    return ssout.str();
}

// Remove chars from begin and end of the passed string, provided the char
// belongs to one of the chars in trimChars.
//
// NOTE: This will work perfectly provided the string being trimmed as well as
// chars being trimmed are made up of NativeCharSize chars from same encoded space.
// For utf-8, this means the ascii equivalent 1byteSized chars of utf8 and not
// variable length MultiNativeCharSize (ie multibye in case of utf-8) ones.
// NOTE: It will also work, if atleast either end of string as well as trimChars
// have NativeCharSize chars from their encoding space, rather than variable
// length MultiNativeCharSize based chars if any. There needs to be NativeCharSized
// chars beyond any chars that get trimmed, on either side.
//
// NOTE: Given the way UTF-8 char encoding is designed, where NativeCharSize 1byte
// encoded chars are fully unique and dont overlap with any bytes from any of the
// variable length MultiNativeCharSize encoded chars in the utf-8 space, so as long as
// the trimChars belong to NativeCharSize chars subset, the logic should work, even
// if string has a mixture of NativeCharSize and MultiNativeCharSize encoded chars.
// Chances are utf-16 and utf-32 also have similar characteristics wrt thier
// NativeCharSize encoded chars (ie those fully encoded within single 16bit and 32bit 
// value respectively), and so equivalent semantic applies to them also.
//
// ALERT: Given that this simple minded logic, works at individual NativeCharSize level
// only, If trimChars involve variable length MultiNativeCharSize encoded chars, then
// * because different NativeCharSize subparts (bytes in case of utf-8) from different
//   MultiNativeCharSize trim chars when clubbed together can map to some other new char
//   in a variable length encoded char space, if there is that new char at either end
//   of the string, it may get trimmed, because of the possibility of mix up mentioned.
// * given that different variable length MultiNativeCharSize encoded chars may have
//   some common NativeCharSize subparts (bytes in case of utf-8) between them, if one
//   of these chars is at either end of the string and another char is in trimChars,
//   then string may get partially trimmed wrt such a char at either end.
//
template <typename TString>
inline TString str_trim_dumb(TString sin, const TString &trimChars=" \t\n") {
#ifdef DUS_DEBUG_VERBOSE
    LOG_TEELN("DBUG:StrTrimDumb:Str:%s", string_as_hex(sin).c_str());
    LOG_TEELN("DBUG:StrTrimDumb:TrimChars:%s", string_as_hex(trimChars).c_str());
#endif
    sin.erase(sin.find_last_not_of(trimChars)+1);
    sin.erase(0, sin.find_first_not_of(trimChars));
    return sin;
}

// Remove chars from begin and end of the passed string, provided the char belongs
// to one of the chars in trimChars.
// NOTE: Internally converts to wchar/wstring to try and support proper trimming,
// wrt possibly more languages, to some extent. IE even if the passed string
// contains multibyte encoded characters in it in utf-8 space (ie MultiNativeCharSize),
// it may get converted to NativeCharSize chars in the expanded wchar_t encoding space,
// thus leading to fixed NativeCharSize driven logic itself handling things sufficiently.
// Look at str_trim_dumb comments for additional aspects.
inline std::string str_trim_oversmart(std::string sIn, const std::string &trimChars=" \t\n") {
    std::wstring wIn;
    mbs_to_wcs(wIn, sIn);
    std::wstring wTrimChars;
    mbs_to_wcs(wTrimChars, trimChars);
    auto wOut = str_trim_dumb(wIn, wTrimChars);
    std::string sOut;
    wcs_to_mbs(sOut, wOut);
    return sOut;
}

// Remove atmost 1 char at the begin and 1 char at the end of the passed string,
// provided the char belongs to one of the chars in trimChars.
//
// NOTE: Chars being trimmed (ie in trimChars) needs to be part of NativeCharSize
// subset of the string's encoded char space, to avoid mix up when working with
// strings which can be utf-8/utf-16/utf-32/sane-variable-length encoded strings.
//
// NOTE:UTF8: This will work provided the string being trimmed as well the chars
// being trimmed are made up of 1byte encoded chars in case of utf8 encoding space.
// If the string being trimmed includes multibyte (ie MultiNativeCharSize) encoded
// characters at either end, then trimming can mess things up, if you have multibyte
// encoded utf-8 chars in the trimChars set.
//
// Currently given that SimpCfg only uses this with NativeCharSize chars in the
// trimChars and most of the platforms are likely to be using utf-8 based char
// space (which is a realtively sane variable length char encoding from this
// logics perspective), so not providing oversmart variant.
//
template <typename TString>
inline TString str_trim_single(TString sin, const TString& trimChars=" \t\n") {
    if (sin.empty()) return sin;
    for(auto c: trimChars) {
        if (c == sin.front()) {
            sin = sin.substr(1, TString::npos);
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

// Convert to lower case, if language has upper and lower case semantic
//
// This works for fixed size encoded char spaces.
//
// For variable length encoded char spaces, it can work
// * if one is doing the conversion for languages which fit into NativeCharSized chars in it
// * AND if one is working with a sane variable length encoding standard
// * ex: this will work if trying to do the conversion for english language within utf-8
//
template <typename TString>
inline TString str_tolower(const TString &sin) {
    TString sout;
    sout.resize(sin.size());
    std::transform(sin.begin(), sin.end(), sout.begin(), [](auto c)->auto {return std::tolower(c);});
#ifdef DUS_DEBUG_VERBOSE
    LOG_TEELN("DBUG:StrToLower:in:%s", string_as_hex(sin).c_str());
    LOG_TEELN("DBUG:StrToLower:out:%s", string_as_hex(sout).c_str());
#endif
    return sout;
}

inline void str_compare_dump(const std::string &s1, const std::string &s2) {
    LOG_TEELN("DBUG:%s:%s:Len:%zu", __func__, s1.c_str(), s1.length());
    LOG_TEELN("DBUG:%s:%s:Len:%zu", __func__, s2.c_str(), s2.length());
    int minLen = s1.length() < s2.length() ? s1.length() : s2.length();
    for(int i=0; i<minLen; i++) {
        LOG_TEELN("DBUG:%s:%d:%c:%c", __func__, i, s1[i], s2[i]);
    }
}


template<typename TypeWithStrSupp>
std::string as_str(TypeWithStrSupp value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}
