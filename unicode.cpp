#include "unicode.h"
#include "unicode-data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <locale>
#include <codecvt>

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

static uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("invalid string");
}

static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cp) {
    std::vector<uint16_t> result;
    if (/* 0x0000 <= cp && */ cp <= 0xffff) {
        result.emplace_back(cp);
    }
    else if (0x10000 <= cp && cp <= 0x10ffff) {
        result.emplace_back(0xd800 | ((cp - 0x10000) >> 10));
        result.emplace_back(0xdc00 | ((cp - 0x10000) & 0x03ff));
    }
    else {
        throw std::invalid_argument("invalid cpt");
    }
    return result;
}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

static uint32_t cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
    assert(offset < utf16.size());
    if (((utf16[0] >> 10) << 10) != 0xd800) {
        auto result = utf16[offset + 0];
        offset += 1;
        return result;
    }

    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
        throw std::invalid_argument("invalid character");
    }

    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
    offset += 2;
    return result;
}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::unordered_map<uint32_t, int> unicode_cpt_type_map() {
    std::unordered_map<uint32_t, int> cpt_types;
    for (auto p : unicode_ranges_digit) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_DIGIT;
        }
    }
    for (auto p : unicode_ranges_letter) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_LETTER;
        }
    }
    for (auto p : unicode_ranges_whitespace) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_WHITESPACE;
        }
    }
    for (auto p : unicode_ranges_accent_mark) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_ACCENT_MARK;
        }
    }
    for (auto p : unicode_ranges_punctuation) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_PUNCTUATION;
        }
    }
    for  (auto p : unicode_ranges_symbol) {
        for (auto i = p.first; i <= p.second; ++i) {
            cpt_types[i] = CODEPOINT_TYPE_SYMBOL;
        }
    }
    for (auto p : unicode_ranges_control) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_CONTROL;
        }
    }
    return cpt_types;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = u'!'; ch <= u'~'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = u'¡'; ch <= u'¬'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = u'®'; ch <= u'ÿ'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = u'!'; ch <= u'~'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = u'¡'; ch <= u'¬'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = u'®'; ch <= u'ÿ'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static inline std::wstring unicode_wstring_from_utf8(const std::string & s) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.from_bytes(s);
}

static inline std::string unicode_wstring_to_utf8(const std::wstring & ws) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(ws);
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string>bpe_encoded_words;
    for (auto word : bpe_words) {
        std::string text_utf = "";
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i)
            text_utf += unicode_cpt_to_utf8(utf_word[i]);

        std::string encoded_token = "";
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

static std::vector<size_t> unicode_gpt2_regex_preprocess(const std::wstring & wtext, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // stroe the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;

    for (auto offset : offsets) {
        const std::string text = unicode_wstring_to_utf8(std::wstring(wtext, start, offset));

        std::string token = "";
        // GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
        bool collecting_numeric = false;
        bool collecting_letter = false;
        bool collecting_special = false;
        bool collecting_whitespace_lookahead = false;
        bool collecting = false;

        std::vector<std::string> text_utf;
        text_utf.reserve(text.size());

        const auto cpts = unicode_cpts_from_utf8(text);
        for (size_t i = 0; i < cpts.size(); ++i) {
            text_utf.emplace_back(unicode_cpt_to_utf8(cpts[i]));
        }

        for (int i = 0; i < (int)text_utf.size(); i++) {
            const std::string & utf_char = text_utf[i];
            bool split_condition = false;
            int bytes_remain = text_utf.size() - i;

            // forward backward lookups
            const std::string & utf_char_next      = (i + 1 < (int)text_utf.size()) ? text_utf[i + 1] : "";
            const std::string & utf_char_next_next = (i + 2 < (int)text_utf.size()) ? text_utf[i + 2] : "";

            // handling contractions
            if (!split_condition && bytes_remain >= 2) {
                // 's|'t|'m|'d
                if (utf_char == "\'" && (utf_char_next == "s" || utf_char_next == "t" || utf_char_next == "m" || utf_char_next == "d")) {
                    split_condition = true;
                }
                if (split_condition) {
                    if (token.size()) {
                        bpe_offsets.emplace_back(unicode_wstring_from_utf8(token).size());
                    }
                    token = utf_char + utf_char_next;
                    bpe_offsets.emplace_back(unicode_wstring_from_utf8(token).size());
                    token = "";
                    i++;
                    continue;
                }
            }
            if (!split_condition && bytes_remain >= 3) {
                // 're|'ve|'ll
                if (utf_char == "\'" && (
                    (utf_char_next == "r" && utf_char_next_next == "e") ||
                    (utf_char_next == "v" && utf_char_next_next == "e") ||
                    (utf_char_next == "l" && utf_char_next_next == "l"))
                    ) {
                    split_condition = true;
                }
                if (split_condition) {
                    // current token + next token can be defined
                    if (token.size()) {
                        bpe_offsets.emplace_back(unicode_wstring_from_utf8(token).size());
                    }
                    token = utf_char + utf_char_next + utf_char_next_next;
                    bpe_offsets.emplace_back(unicode_wstring_from_utf8(token).size());
                    token = "";
                    i += 2;
                    continue;
                }
            }

            if (!split_condition && !collecting) {
                if (unicode_cpt_type(utf_char) == CODEPOINT_TYPE_LETTER || (!token.size() && utf_char == " " && unicode_cpt_type(utf_char_next) == CODEPOINT_TYPE_LETTER)) {
                    collecting_letter = true;
                    collecting = true;
                }
                else if (unicode_cpt_type(utf_char) == CODEPOINT_TYPE_DIGIT || (!token.size() && utf_char == " " && unicode_cpt_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    collecting_numeric = true;
                    collecting = true;
                }
                else if (
                    ((unicode_cpt_type(utf_char) != CODEPOINT_TYPE_LETTER && unicode_cpt_type(utf_char) != CODEPOINT_TYPE_DIGIT) && (unicode_cpt_type(utf_char) != CODEPOINT_TYPE_WHITESPACE)) ||
                    (!token.size() && utf_char == " " && unicode_cpt_type(utf_char_next) != CODEPOINT_TYPE_LETTER && unicode_cpt_type(utf_char_next) != CODEPOINT_TYPE_DIGIT && unicode_cpt_type(utf_char_next) != CODEPOINT_TYPE_WHITESPACE)
                    ) {
                    collecting_special = true;
                    collecting = true;
                }
                else if (unicode_cpt_type(utf_char) == CODEPOINT_TYPE_WHITESPACE && unicode_cpt_type(utf_char_next) == CODEPOINT_TYPE_WHITESPACE) {
                    collecting_whitespace_lookahead = true;
                    collecting = true;
                }
                else if (unicode_cpt_type(utf_char) == CODEPOINT_TYPE_WHITESPACE) {
                    split_condition = true;
                }
            }
            else if (!split_condition && collecting) {
                if (collecting_letter && unicode_cpt_type(utf_char) != CODEPOINT_TYPE_LETTER) {
                    split_condition = true;
                }
                else if (collecting_numeric && unicode_cpt_type(utf_char) != CODEPOINT_TYPE_DIGIT) {
                    split_condition = true;
                }
                else if (collecting_special && (unicode_cpt_type(utf_char) == CODEPOINT_TYPE_LETTER || unicode_cpt_type(utf_char) == CODEPOINT_TYPE_DIGIT || unicode_cpt_type(utf_char) == CODEPOINT_TYPE_WHITESPACE)) {
                    split_condition = true;
                }
                else if (collecting_whitespace_lookahead && (unicode_cpt_type(utf_char_next) == CODEPOINT_TYPE_LETTER || unicode_cpt_type(utf_char_next) == CODEPOINT_TYPE_DIGIT)) {
                    split_condition = true;
                }
            }

            if (utf_char_next == "") {
                split_condition = true; // final
                token += utf_char;
            }

            if (split_condition) {
                if (token.size()) {
                    bpe_offsets.emplace_back(unicode_wstring_from_utf8(token).size());
                }
                token = utf_char;
                collecting = false;
                collecting_letter = false;
                collecting_numeric = false;
                collecting_special = false;
                collecting_whitespace_lookahead = false;
            }
            else {
                token += utf_char;
            }
        }

        start += offset;
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_preprocess(const std::wstring & text, const std::vector<size_t> & offsets, const std::wstring & regex_expr) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // stroe the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::wcmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

static bool unicode_regex_equivalent_wregex_exists(const std::string & regex) {
    return unicode_regex_equivalent_wregex.find(regex) != unicode_regex_equivalent_wregex.end();
}

static bool unicode_regex_with_custom_preprocessor_exists(const std::string & regex) {
    return unicode_regex_with_custom_preprocessor.find(regex) != unicode_regex_with_custom_preprocessor.end();
}

static std::vector<size_t> unicode_regex_custom_preprocess(const std::string & regex, const std::wstring & wtext, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_gpt2_regex_preprocess(wtext, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cp) {
    std::string result;
    if (/* 0x00 <= cp && */ cp <= 0x7f) {
        result.push_back(cp);
    }
    else if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else {
        throw std::invalid_argument("invalid codepoint");
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    std::vector<uint32_t> result;
    result.reserve(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        auto it = unicode_map_nfd.find(cpts[i]);
        if (it == unicode_map_nfd.end()) {
            result.push_back(cpts[i]);
        } else {
            result.push_back(it->second);
        }
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}

int unicode_cpt_type(uint32_t cp) {
    static std::unordered_map<uint32_t, int> cpt_types = unicode_cpt_type_map();
    const auto it = cpt_types.find(cp);
    return it == cpt_types.end() ? CODEPOINT_TYPE_UNIDENTIFIED : it->second;
}

int unicode_cpt_type(const std::string & utf8) {
    if (utf8.length() == 0) {
        return CODEPOINT_TYPE_UNIDENTIFIED;
    }
    size_t offset = 0;
    return unicode_cpt_type(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

char32_t unicode_tolower(char32_t cp) {
    auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    std::wstring wtext = unicode_wstring_from_utf8(text);

    std::vector<size_t> bpe_offsets = {wtext.size()};

    for (auto & regex_expr : regex_exprs) {
        if (unicode_regex_equivalent_wregex_exists(regex_expr)) {
            const std::wstring & wregex_expr = unicode_regex_equivalent_wregex.at(regex_expr);
            bpe_offsets = unicode_regex_preprocess(wtext, bpe_offsets, wregex_expr);
        } else if (unicode_regex_with_custom_preprocessor_exists(regex_expr)) {
            bpe_offsets = unicode_regex_custom_preprocess(regex_expr, wtext, bpe_offsets);
        } else {
            throw std::runtime_error("Unicode regex is not found");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back(unicode_wstring_to_utf8(std::wstring(wtext, start, offset)));
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
