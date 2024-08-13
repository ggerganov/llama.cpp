#if defined(_MSC_VER)
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#include "ggml.h"
#include "unicode.h"
#include "unicode-data.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <locale>
#include <codecvt>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
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
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cp) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cp && */ cp <= 0xffff) {
//        result.emplace_back(cp);
//        return result;
//    }
//    if (0x10000 <= cp && cp <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cp - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cp - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
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
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
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

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        GGML_ASSERT(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        static const codepoint_categ SENTINEL = codepoint_categ::UNDEF + 1;
        auto _get_categ = [&] (const size_t pos) -> codepoint_categ {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_category(cpts[pos]) : SENTINEL;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            GGML_ASSERT(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto categ = _get_categ(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = _get_cpt(pos+1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = _get_cpt(pos+2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            auto categ2 = (cpt == ' ' ? _get_categ(pos+1) : categ);
            // regex: <space>?\p{L}+
            if (categ2.is_L()) {
                pos += (cpt == ' ');
                while (categ2.is_L()) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (categ2.is_N()) {
                pos += (cpt == ' ');
                while (categ2.is_N()) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                pos += (cpt == ' ');
                while (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                    categ2 = _get_categ(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (_get_categ(pos+num_whitespaces).is_whitespace()) {
                num_whitespaces++;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        GGML_ASSERT(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        static const codepoint_categ SENTINEL = codepoint_categ::UNDEF + 1;
        auto _get_categ = [&] (const size_t pos) -> codepoint_categ {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_category(cpts[pos]) : SENTINEL;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            GGML_ASSERT(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto categ = _get_categ(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || categ.is_N())) {
                if (categ.is_L() || _get_categ(pos+1).is_L()) {  // one or more letters
                    pos++;
                    while (_get_categ(pos).is_L()) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (categ.is_N()) {
                size_t ini = pos;
                while (_get_categ(pos).is_N()) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto categ2 = (cpt == ' ' ? _get_categ(pos+1) : categ);
            if (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                pos += (cpt == ' ');
                while (!(categ2.is_whitespace() | categ2.is_L() | categ2.is_N()) && categ2 != SENTINEL) {
                    categ2 = _get_categ(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_categ(pos+num_whitespaces).is_whitespace()) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// use std::wregex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::wstring & wtext, const std::wstring & regex_expr, const std::vector<size_t> & offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
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

// use std::regex to split the text
static std::vector<size_t> unicode_regex_split_stl(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            std::cmatch match = *it;
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

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {

        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
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
        return result;
    }
    if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.cbegin(), unicode_ranges_nfd.cend(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}

codepoint_categ unicode_cpt_category(const uint32_t cp) {
    static const std::vector<codepoint_categ> cpt_categs = [] {
        std::vector<codepoint_categ> cpt_categs(MAX_CODEPOINTS, codepoint_categ::UNDEF);
        uint32_t cpt = 0;
        for (uint16_t rle : unicode_rle_codepoints_categs) {
            const uint32_t index = rle & 31;
            const uint32_t count = rle >> 5;
            auto categ = codepoint_categ::from_index(index);
            //printf("Codepoints 0x%05X to 0x%05X categ %s\n", cpt, cpt + count, categ.c_str());
            categ.set_flag(codepoint_categ::DIGITS, categ.is_Nd());               // \d --> \p{Nd}
            categ.set_flag(codepoint_categ::WORDS, categ.is_L() | categ.is_N());  // \w --> \p{L} \p{N} _
            for (uint32_t i = 0; i <= count; ++i) {
                cpt_categs[cpt++] = categ;
            }
        }
        GGML_ASSERT(cpt == MAX_CODEPOINTS);

        cpt_categs['_'].set_flag(codepoint_categ::WORDS);  // \w --> \p{L} \p{N} _

        for (auto p : unicode_ranges_whitespace) {
            for (uint32_t cpt = p.first; cpt <= p.second; ++cpt) {
                cpt_categs[cpt].set_flag(codepoint_categ::WHITESPACES);
            }
        }

        //for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        //    cpt_categs[cpt].set_flag(codepoint_categ::NORM_NFD);
        //}

        return cpt_categs;
    }();
    return cp < cpt_categs.size() ? cpt_categs[cp] : codepoint_categ{};
}

codepoint_categ unicode_cpt_category(const std::string & utf8) {
    if (utf8.empty()) {
        return codepoint_categ{};  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_category(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cp) {
    auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    // std::wregex does not support unicode categories: \p{N}, \p{L}, \p{Lu}, \p{Ll} ...
    // std::wregex does not support unicode whitespaces \s: 0x85, 0xA0, 0x001680 ... 0x003000.
    // std::wregex allows full wchar_t 32 bit codepoints, not limited to standard max 0x110000.
    // The main idea is to insert unicode category bits into all regex and text codepoints.
    //   Max unicode codepoint 0x110000 fits in 21 bits.
    //   Store unicode category and subcategory in 10 bits.
    //   Set the high bit to zero to keep wchar_t positive (uint32_t codepoints).
    //   Categorized codepoint:
    //     1 bit zero + 7 bits category + 3 bits subcategory index + 21 bits codepoint
    //     0b0'XXXXXXX'xxx'ccccccccccccccccccccc
    // A "categorized codepoint" re-defines the ordering keeping category hierarchy.
    //   All high category codepoints \p{X} fall into the range:
    //     0b0'XXXXXXX'000'000000000000000000000
    //     0b0'XXXXXXX'111'111111111111111111111
    //   All subcategory codepoints \p{Xx} fall into the range:
    //     0b0'XXXXXXX'xxx'000000000000000000000
    //     0b0'XXXXXXX'xxx'111111111111111111111
    // Processing steps:
    //   Build a lists of "categorized codepoints/ranges" for replacing regex \s \w and \d.
    //   Replace all regex codepoints/ranges with respective "categorized codepoints/ranges".
    //   Replace all text codepoints with respective "categorized codepoints".
    // Caveats:
    //   Some regex ranges starts and ends with different category/subcategory.
    //   Split the ranges in sub-ranges to ensure a single category to maintain the new hierarchy.
    //   This forces iterating all ranges and could produce long sub-range sequences.

    //TODO: Regex processing can be cached.

    // insert unicode category and subcategory before codepoint bits
    // 1 bit zero + 7 bits category + 3 bits subcategory index + 21 bits zero
    static const auto categorized_prefix = [] (const codepoint_categ categ) -> wchar_t {
        static const uint32_t MASK    = codepoint_categ::MASK;  // category mask
        static const uint32_t SUBMASK = codepoint_categ::SUBMASK & ~codepoint_categ::MASK;  // subcategory mask
        return (wchar_t) (((categ.encoded & MASK) << (21+3)) | ((categ.encoded & SUBMASK) << (21-7)));
    };

    // insert unicode category and subcategory before codepoint bits
    // 1 bit zero + 7 bits category + 3 bits subcategory index + 21 bits codepoint
    static const auto categorize_codepoint = [] (const uint32_t cpt) -> wchar_t {
        GGML_ASSERT(cpt < (1 << 21));
        return categorized_prefix(unicode_cpt_category(cpt)) | (wchar_t)cpt;
    };

    // remove the categorized prefix bits and restore original codepoint bits
    static const auto decategorize_codepoint = [] (const wchar_t cpt) -> uint32_t {
        return (uint32_t) cpt & ((1 << 21) - 1);
    };

    // returns the respective categorized codepoint range of the category/subcategory
    static const auto categorize_range_from_chars = [] (const char categ, const char subcateg) {
        const wchar_t range_ini = categorized_prefix(codepoint_categ::from_chars(categ, subcateg));
        const wchar_t range_end = (wchar_t) (range_ini | (subcateg ? (1<<21)-1 : (1<<24)-1));
        return std::pair<wchar_t, wchar_t>(range_ini, range_end);
    };

    // helper function to append/concat regex expressions
    auto wregex_append_subregex = [] (std::wstring & wregex, const std::wstring & subregex, const bool add_squares, const bool negated) {
        if (add_squares) {
            wregex += '[';
            if (negated) {
                wregex += '^';
            }
            wregex += subregex;
            wregex += ']';
        } else {
            GGML_ASSERT(!negated);  //TODO: negation inside square brackets: \S \W \D
            wregex += subregex;
        }
    };

    // \d digits replacement
    static const std::wstring wregex_digits = {
        categorize_codepoint('0'), '-', categorize_codepoint('9'),
    };

    // \w words replacement
    static const std::wstring wregex_words = {
        categorize_codepoint('_'),
        categorize_codepoint('0'), '-', categorize_codepoint('9'),
        categorize_codepoint('A'), '-', categorize_codepoint('Z'),
        categorize_codepoint('a'), '-', categorize_codepoint('z'),
    };

    // \s whitespaces replacement
    static const std::wstring wregex_whitespaces = [] {
        std::wstring wregex_whitespaces;
        for (const auto & range : unicode_ranges_whitespace) {
            wregex_whitespaces += categorize_codepoint(range.first);
            if (range.second > range.first) {
                wregex_whitespaces += '-';
                wregex_whitespaces += categorize_codepoint(range.second);
            }
        }
        return wregex_whitespaces;
    }();

    GGML_ASSERT(sizeof(wchar_t) == sizeof(uint32_t));
    std::wstring wtext = unicode_wstring_from_utf8(text);

    std::vector<size_t> offsets = { wtext.size() };

    for (auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, offsets);

        if (!tmp.empty()) {
            offsets = std::move(tmp);
            continue;
        }

        std::wstring wregex;
        bool inside_square = false;
        bool is_cpt_range  = false;

        const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
        wregex.reserve(2 * cpts_regex.size());

        for (size_t i = 0; i < cpts_regex.size(); ++i) {
            uint32_t cpt = cpts_regex[i];

            // parse regex metacharacters
            wregex += (wchar_t) cpt;
            if (inside_square) {
                switch(cpt) {
                    case '^':
                        if (cpts_regex[i - 1] != '[') {
                            break;
                        }
                        continue;
                    case ']':
                        inside_square = false;
                        continue;
                    case '-':
                        is_cpt_range = true;
                        continue;
                }
            } else {
                switch(cpt) {
                    case '^':
                        if (i > 0) {
                            break;
                        }
                        continue;
                    case '$':
                        if (i + 1 < cpts_regex.size()) {
                            break;
                        }
                        continue;
                    case '[':
                        inside_square = true;
                        continue;
                    case '{':
                        while (cpt && cpt != '}') {
                            cpt = cpts_regex[++i];
                            wregex += (wchar_t) cpt;
                        }
                        continue;
                    case '}':
                    case ']':
                        GGML_ABORT("invalid regex");
                    case '(':
                        if (cpts_regex[i + 1] == '?') {  // (?: (?i: (?= (?! (?<= (?<!
                            if (cpts_regex[i + 2] == ':') {
                                wregex += (wchar_t) cpts_regex[++i];
                                wregex += (wchar_t) cpts_regex[++i];
                            } else if (cpts_regex[i + 2] == 'i') {
                                wregex += (wchar_t) cpts_regex[++i];
                                wregex += (wchar_t) cpts_regex[++i];
                                wregex += (wchar_t) cpts_regex[++i];
                                GGML_ASSERT(cpts_regex[i] == ':');
                            } else {
                                wregex += (wchar_t) cpts_regex[++i];
                                wregex += (wchar_t) cpts_regex[++i];
                                if (cpts_regex[i] == '<') {
                                    wregex += (wchar_t) cpts_regex[++i];
                                }
                                GGML_ASSERT(cpts_regex[i] == '=' || cpts_regex[i] == '!');
                            }
                        }
                        continue;
                    case ')':
                    case '|':
                    case '.':
                    case '?':
                    case '+':
                    case '*':
                        continue;
                }
            }
            wregex.pop_back();

            // parse unicode categories and subcategories, replace category with the categorized range
            if (cpt == '\\' && cpts_regex[i + 1] == 'p' && cpts_regex[i + 2] == '{') {
                GGML_ASSERT(cpts_regex[i + 3] && cpts_regex[i + 4]);
                std::pair<wchar_t, wchar_t> range;
                if (cpts_regex[i + 4] == '}') {
                    range = categorize_range_from_chars((char)cpts_regex[i + 3], (char)'\0');
                    i += 4;
                } else {
                    range = categorize_range_from_chars((char)cpts_regex[i + 3], (char)cpts_regex[i + 4]);
                    i += 5;
                }
                GGML_ASSERT(cpts_regex[i] == '}');
                const std::wstring subregex = {range.first, '-', range.second};
                wregex_append_subregex(wregex, subregex, !inside_square, false);
                continue;
            }

            // parse more metcharacters and espaped characters
            if (cpt == '\\') {
                switch (cpts_regex[i + 1]) {
                    case 's':  // \s whitespaces
                    case 'S':  // \S no whitespaces
                        wregex_append_subregex(wregex, wregex_whitespaces, !inside_square, cpts_regex[++i] == 'S');
                        continue;
                    case 'w':  // \w words
                    case 'W':  // \W no words
                        wregex_append_subregex(wregex, wregex_words, !inside_square, cpts_regex[++i] == 'W');
                        continue;
                    case 'd':  // \d digits
                    case 'D':  // \D no digits
                        wregex_append_subregex(wregex, wregex_digits, !inside_square, cpts_regex[++i] == 'D');
                        continue;
                    case 't':  ++i;  cpt = '\t';  break;
                    case 'r':  ++i;  cpt = '\r';  break;
                    case 'n':  ++i;  cpt = '\n';  break;
                    case 'x':  GGML_ABORT("TODO");  //TODO: hex values
                    case 'u':  GGML_ABORT("TODO");  //TODO: unicode values
                    case 'U':  GGML_ABORT("TODO");  //TODO: unicode values
                    default:  // escaped character
                        GGML_ASSERT(!is_cpt_range);
                        cpt = cpts_regex[++i];
                        GGML_ASSERT(cpt < 0x80);
                        break;
                }
            }

            if (is_cpt_range) {
                // Some regex ranges starts and ends with different category/subcategory.
                // Split the ranges in sub-ranges to ensure a single category to maintain the new hierarchy.
                // Warning: This forces iterating all ranges and could produce long sub-range sequences.
                GGML_ASSERT(wregex.size() && wregex.back() == '-');
                wregex.pop_back();
                wchar_t categorized = wregex.back();
                uint32_t range_ini = decategorize_codepoint(categorized);
                const uint32_t range_end = cpt;
                GGML_ASSERT(range_ini <= range_end);
                codepoint_categ range_categ = unicode_cpt_category(range_ini);
                for (cpt = range_ini + 1; cpt <= range_end; ++cpt) {
                    codepoint_categ categ = unicode_cpt_category(cpt);
                    if (categ == range_categ) {  // still same range category ?
                        ++categorized;
                        if (cpt == range_ini + 1) {  // single step, no need range
                            wregex += categorized;
                        } else if (cpt == range_ini + 2) {  // need range if +2 step
                            wregex.back() = '-';
                            wregex += categorized;
                        } else {
                            wregex.back() = categorized;  // keep range growing
                        }
                    } else {  // new range category
                        categorized = categorize_codepoint(cpt);
                        wregex += categorized;
                        range_categ = categ;
                        range_ini = cpt;
                    }
                }
                is_cpt_range = false;
            } else {
                wregex += categorize_codepoint(cpt);
            }
        }

        // categorize all wtext codepoints
        if (wtext.size() && wtext[0] < MAX_CODEPOINTS) {  // if not already categorized
            for (size_t i = 0; i < wtext.size(); ++i) {
                wtext[i] = categorize_codepoint((uint32_t) wtext[i]);
            }
        }

        offsets = unicode_regex_split_stl(wtext, wregex, offsets);
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            const uint32_t cpt = decategorize_codepoint(wtext[i]);
            bpe_words.back() += unicode_cpt_to_utf8(cpt);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
