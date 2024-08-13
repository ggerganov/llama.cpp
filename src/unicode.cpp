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

// Custom std::regex specializations for 32bit unicode codepoints
//   std::wregex does not support unicode categories: \p{N}, \p{L}, \p{Lu}, \p{Ll} ...
//   std::wregex does not support unicode whitespaces \s: 0x85, 0xA0, 0x001680 ... 0x003000.
//   std::wregex supports full 32 bit codepoints, not limited to standard max 0x110000.
namespace std {
    using codepoint = uint32_t;  // codepoint type for all template specializations

    // Minimal required implementation for std::regex string processing
    template<>  // custom specialized std::ctype<codepoint>
    class ctype<codepoint> {
        public:

        using CharT = codepoint;
        using char_type = CharT;

        using mask = uint8_t;          //NOTE: see std::ctype_base
        static const mask digit  = 1;  // requiered variable names
        static const mask xdigit = 2;  // user defined values
        static const mask alpha  = 3;  // used to be a bitmask
        static const mask upper  = 4;  // we do not need a bitmask
        static const mask lower  = 5;  // using a sequence instead

        static locale::id id;  // required by std::locale::facet

        bool is(mask m, char_type c) const {
            switch (m) {
                case digit:  return ('0' <= c && c <= '9');
                case xdigit: return ('0' <= c && c <= '9') || ('A' <= c && c <= 'F');
                case alpha:  return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z');
                case upper:  return ('A' <= c && c <= 'Z');
                case lower:  return ('a' <= c && c <= 'z');
                default:     return false;
            }
        }

        char_type toupper(char_type c) const {
            return ('a' <= c && c <= 'z') ? c - ('a' - 'A') : c;
        }

        char_type tolower(char_type c) const {
            return ('A' <= c && c <= 'Z') ? c + ('a' - 'A') : c;
        }

        char_type widen(char c) const {  // char to codepoint
            return (char_type) c;
        }

        char narrow(char_type c, char dfault) const {  // codepoint to char
            return (c < 0x80 ? (char)c : dfault);
        }
    };

    locale::id ctype<codepoint>::id = {};

    template<>  // specialization to use our custom specialized std::ctype<codepoint>
    const std::ctype<codepoint> & use_facet<std::ctype<codepoint>>(const std::locale &) {
        static std::ctype<codepoint> ctype_uint32 = {};
        return ctype_uint32;
    }

    template<>  // specialization to use our custom specialized std::ctype<codepoint>
    const std::ctype<codepoint> & use_facet<const std::ctype<codepoint>>(const std::locale & loc) {
        return use_facet<std::ctype<codepoint>>(loc);
    }

    // Minimal required implementation for std::regex string processing
    template<>  // custom specialized std::regex_traits<codepoint>
    class regex_traits<codepoint> {
    public:

        using CharT       = codepoint;
        using char_type   = codepoint;
        using size_type   = size_t;
        using string_type = std::basic_string<CharT>;
        using locale_type = std::locale;
        using char_class_type = uint64_t;

        #if (defined(_WIN32) || defined(_WIN64))  // MSVC class _Regex_traits
            using _Uelem = CharT;
            static const auto _Ch_upper = std::ctype<CharT>::upper;
            static const auto _Ch_alpha = std::ctype<CharT>::alpha;
        #endif

        CharT translate(CharT c) const {
            return c;
        }

        CharT translate_nocase(CharT c) const {
            return unicode_tolower(c);
        }

        template<typename It>
        string_type transform(It first, It last) const {
            GGML_ASSERT(false);   //TODO: not needed ?
            return {first, last}; //TODO: not tested
        }

        template<typename It>
        string_type transform_primary(It first, It last) const {
            (void) first;
            (void) last;
            GGML_ASSERT(*first < MAX_CODEPOINTS);  // valid codepoint
            return {};
        }

        template<typename It>
        string_type lookup_collatename(It first, It last) const {
            (void) last;
            GGML_ASSERT(*first & (1 << 31));
            return {*first};
        }

        template<typename It>
        char_class_type lookup_classname(It first, It last, bool icase = false) const {
            (void) last;
            (void) icase;
            const uint32_t encoded = *first;
            codepoint_categ categ = {};
            switch(encoded) {
                case 's':
                case 'S':  // negation is internally tracked
                    categ.set_flag(codepoint_categ::WHITESPACES);
                    return categ.expand_bits();
                case 'w':
                case 'W':  // negation is internally tracked
                    categ.set_flag(codepoint_categ::WORDS);
                    return categ.expand_bits();
                case 'd':
                case 'D':  // negation is internally tracked
                    categ.set_flag(codepoint_categ::DIGITS);
                    return categ.expand_bits();
                default: {  // unicode category \p{Xx} encoded in codepoint
                    GGML_ASSERT(encoded & (1 << 31));  // make sure its our custom codepoint encoding the category
                    const bool negated = encoded & (1 << 30);  // negation of 'character class expression' are not internally tracked
                    categ = {(uint16_t) encoded};
                    return ((uint64_t) negated << 63) | categ.expand_bits(false);
                }
            }
        }

        bool isctype(CharT c, char_class_type mask) const {
            const bool negated = mask & (1llu << 63);
            mask &= unicode_cpt_category(c).expand_bits();
            return negated ^ (bool) mask;
        }

        int value(CharT c, int radix) const {  // char to int value
            switch (radix) {
                case 8:  return ('0' <= c && c <= '7') ? (int)c - '0' : -1;
                case 10: return ('0' <= c && c <= '9') ? (int)c - '0' : -1;
                case 16: return ('0' <= c && c <= '9') ? (int)c - '0' : (('A' <= c && c <= 'F') ? (int)c - 'A' + 10 : -1);
                default: return -1;
            }
        }

        const locale_type & imbue(const locale_type &) {  // set locale  //NOTE: ignoring locales
            return std::locale::classic();
        }

        const locale_type & getloc() const {  // get locale  //NOTE: ignoring locales
            return std::locale::classic();
        }
    };
}

static std::vector<uint32_t> unicode_regex_prepare(const std::string & regex) {
    std::vector<uint32_t> regex_cpts;
    regex_cpts.reserve(regex.size() * 12 / 10);  // estimate +20%

    size_t offset = 0;
    int inside_square = 0;
    bool any_positive = false;
    bool any_negative = false;

    const size_t size = regex.size();
    while (offset < size) {
        inside_square += regex[offset] == '[';
        inside_square -= regex[offset] == ']';
        GGML_ASSERT(inside_square >= 0);
        if (!inside_square) {
            any_positive = false;
            any_negative = false;
        }

        if (regex[offset] == '\\') {
            const size_t i = offset + 1;
            if (regex[i] == 'p' || regex[i] == 'P') {
                // convert \p{Xx} to custom 'character class expression' [:Xy:]
                if (regex[i + 1] == '{' && regex[i + 2] && regex[i + 3]) {
                    codepoint_categ categ = {};
                    if (regex[i + 3] == '}') {
                        categ = codepoint_categ::from_chars(regex[i + 2]);
                        offset += 5;
                    } else if (regex[i + 3] != '}' && regex[i + 4] == '}') {
                        categ = codepoint_categ::from_chars(regex[i + 2], regex[i + 3]);
                        offset += 6;
                    }
                    bool negated = regex[i] == 'P';
                    any_positive |= !negated;
                    any_negative |= negated;
                    GGML_ASSERT(any_positive != any_negative);  //BUG: can not mix 'p' and 'P' inside []
                    GGML_ASSERT(sizeof(categ) <= 2);
                    // encoded category in 32 bits codepoint
                    uint32_t cpt_categ = (1 << 31) | (negated << 30) | categ.encoded;
                    if (inside_square) {
                        regex_cpts.insert(regex_cpts.end(), {'[', ':', cpt_categ, ':', ']'});
                    } else {
                        regex_cpts.insert(regex_cpts.end(), {'[', '[', ':', cpt_categ, ':', ']', ']'});
                    }
                    continue;
                }
            }
        }

        regex_cpts.push_back(unicode_cpt_from_utf8(regex, offset));
    }

    return regex_cpts;
}

// use std::basic_regex<uint32_t> to split the text codepoints
static std::vector<size_t> unicode_regex_split_stl(const std::vector<uint32_t> & text_cpts, const std::vector<uint32_t> & regex_cpts, const std::vector<size_t> & offsets) {
    using regex_type = std::basic_regex<uint32_t>;
    using iter_type = std::regex_iterator<const uint32_t *>;
    regex_type regex(regex_cpts.begin(), regex_cpts.end());
    const iter_type end;

    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // reserve memory for the approximate size
    const uint32_t * text_data = text_cpts.data();
    for (auto offset : offsets) {
        iter_type it(text_data, text_data + offset, regex);
        int64_t start_idx = 0;
        while (it != end) {
            if (it->position() > start_idx) {
                bpe_offsets.emplace_back(it->position() - start_idx);
            }
            bpe_offsets.emplace_back(it->length());
            start_idx = it->position() + it->length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        text_data += offset;
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

std::vector<std::string> unicode_regex_split(const std::string & text_utf8, const std::vector<std::string> & regex_exprs) {
    const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text_utf8);
    std::vector<size_t> offsets = { cpts.size() };

    for (auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text_utf8, regex_expr, offsets);

        if (!tmp.empty()) {
            offsets = std::move(tmp);
            continue;
        }

        const auto regex_cpts = unicode_regex_prepare(regex_expr);
        offsets = unicode_regex_split_stl(cpts, regex_cpts, offsets);
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
