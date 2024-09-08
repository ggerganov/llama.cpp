#pragma once

#include <cstdint>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>

struct codepoint_categ {
    // 0bffffff'ccccccc'sss --> 6 bits flags + 7 bits category + 3 bits subcategory
    enum _category : uint16_t {
        UNDEF = 0,         // \p{Cn} Undefined
        C = 1 << (0 + 3),  // \p{C}  Control
        L = 1 << (1 + 3),  // \p{L}  Letter
        M = 1 << (2 + 3),  // \p{M}  Mark
        N = 1 << (3 + 3),  // \p{N}  Number
        P = 1 << (4 + 3),  // \p{P}  Punctuation
        S = 1 << (5 + 3),  // \p{S}  Symbol
        Z = 1 << (6 + 3),  // \p{Z}  Separator
        Cc = C | 1,  // \p{Cc} Control
        Cf = C | 2,  // \p{Cf} Format
        Co = C | 3,  // \p{Co} Private Use
        Cs = C | 4,  // \p{Cs} Surrrogate
        Ll = L | 1,  // \p{Ll} Lowercase Letter
        Lm = L | 2,  // \p{Lm} Modifier Letter
        Lo = L | 3,  // \p{Lo} Other Letter
        Lt = L | 4,  // \p{Lt} Titlecase Letter
        Lu = L | 5,  // \p{Lu} Uppercase Letter
        Mc = M | 1,  // \p{Mc} Spacing Mark
        Me = M | 2,  // \p{Me} Enclosing Mark
        Mn = M | 3,  // \p{Mn} Nonspacing Mark
        Nd = N | 1,  // \p{Nd} Decimal Number
        Nl = N | 2,  // \p{Nl} Letter Number
        No = N | 3,  // \p{No} Other Number
        Pc = P | 1,  // \p{Pc} Connector Punctuation
        Pd = P | 2,  // \p{Pd} Dash Punctuation
        Pe = P | 3,  // \p{Pe} Close Punctuation
        Pf = P | 4,  // \p{Pf} Final Punctuation
        Pi = P | 5,  // \p{Pi} Initial Punctuation
        Po = P | 6,  // \p{Po} Other Punctuation
        Ps = P | 7,  // \p{Ps} Open Punctuation
        Sc = S | 1,  // \p{Sc} Currency Symbol
        Sk = S | 2,  // \p{Sk} Modifier Symbol
        Sm = S | 3,  // \p{Sm} Math Symbol
        So = S | 4,  // \p{So} Other Symbol
        Zl = Z | 1,  // \p{Zl} Line Separator
        Zp = Z | 2,  // \p{Zp} Paragraph Separator
        Zs = Z | 3,  // \p{Zs} Space Separator
        SUBMASK = (1 <<  3) - 1,  // 3 bits   0b000000'0000000'111
        MASK    = (1 << 10) - 1,  // 7+3 bits 0b000000'1111111'111
    };

    enum _flags : uint16_t {
        WHITESPACES = (1 << 10),  // regex: \s
        WORDS       = (1 << 11),  // regex: \w
        DIGITS      = (1 << 12),  // regex: \d
        //Norm NFD/NFC  = ...,
    };

    inline codepoint_categ(const uint16_t categ=0) : encoded{categ} {}

    inline void set_flag(_flags flags, bool value = true) {
        flags = (_flags) (flags & ~MASK);  // do not modify category bits
        encoded = value ? (encoded | flags) : (encoded & ~flags);
    }

    inline uint16_t get_category() const { return encoded & MASK; }

    inline bool is_undefined() const { return !encoded; }
    inline bool is_defined() const { return encoded; }

    inline uint16_t is_whitespace() const { return encoded & WHITESPACES; }
    inline uint16_t is_word()       const { return encoded & WORDS;  }
    inline uint16_t is_digit()      const { return encoded & DIGITS; }

    inline uint16_t is_C() const { return encoded & C; }
    inline uint16_t is_L() const { return encoded & L; }
    inline uint16_t is_M() const { return encoded & M; }
    inline uint16_t is_N() const { return encoded & N; }
    inline uint16_t is_P() const { return encoded & P; }
    inline uint16_t is_S() const { return encoded & S; }
    inline uint16_t is_Z() const { return encoded & Z; }

    inline bool is_Cc() const { return (encoded & MASK) == Cc; }
    inline bool is_Cf() const { return (encoded & MASK) == Cf; }
    inline bool is_Co() const { return (encoded & MASK) == Co; }
    inline bool is_Cs() const { return (encoded & MASK) == Cs; }
    inline bool is_Ll() const { return (encoded & MASK) == Ll; }
    inline bool is_Lm() const { return (encoded & MASK) == Lm; }
    inline bool is_Lo() const { return (encoded & MASK) == Lo; }
    inline bool is_Lt() const { return (encoded & MASK) == Lt; }
    inline bool is_Lu() const { return (encoded & MASK) == Lu; }
    inline bool is_Mc() const { return (encoded & MASK) == Mc; }
    inline bool is_Me() const { return (encoded & MASK) == Me; }
    inline bool is_Mn() const { return (encoded & MASK) == Mn; }
    inline bool is_Nd() const { return (encoded & MASK) == Nd; }
    inline bool is_Nl() const { return (encoded & MASK) == Nl; }
    inline bool is_No() const { return (encoded & MASK) == No; }
    inline bool is_Pc() const { return (encoded & MASK) == Pc; }
    inline bool is_Pd() const { return (encoded & MASK) == Pd; }
    inline bool is_Pe() const { return (encoded & MASK) == Pe; }
    inline bool is_Pf() const { return (encoded & MASK) == Pf; }
    inline bool is_Pi() const { return (encoded & MASK) == Pi; }
    inline bool is_Po() const { return (encoded & MASK) == Po; }
    inline bool is_Ps() const { return (encoded & MASK) == Ps; }
    inline bool is_Sc() const { return (encoded & MASK) == Sc; }
    inline bool is_Sk() const { return (encoded & MASK) == Sk; }
    inline bool is_Sm() const { return (encoded & MASK) == Sm; }
    inline bool is_So() const { return (encoded & MASK) == So; }
    inline bool is_Zl() const { return (encoded & MASK) == Zl; }
    inline bool is_Zp() const { return (encoded & MASK) == Zp; }
    inline bool is_Zs() const { return (encoded & MASK) == Zs; }

    inline uint64_t expand_bits(const bool add_categ=true) const {  // one bit for each category/subcateory and flags
        const uint32_t subindex = encoded & SUBMASK;
        const uint64_t bits = (encoded & MASK) >> 3;
        const uint64_t flags = encoded >> 10;
        return (flags << (7 * 8)) | (bits << (7 * subindex)) | (bits * add_categ);
    }

    inline bool is_in_range(const codepoint_categ other) const {  // this.first <= other <= this.last
        if (encoded & SUBMASK) {
            return encoded == other.encoded;  // no range
        }
        if (encoded & MASK) {
            return encoded == (other.encoded & ~SUBMASK);  // from 0bffffff'ccccccc'000 to 0bffffff'ccccccc'111
        }
        return encoded == (other.encoded & ~MASK);  // from 0bffffff'0000000'000 to 0bffffff'1111111'111
    }

    inline bool operator == (const codepoint_categ other) const {
        return encoded == other.encoded;
    }

    inline bool operator != (const codepoint_categ other) const {
        return encoded != other.encoded;
    }

    const char * c_str() const {
        static const std::map<uint16_t, const char *> map = {
            {UNDEF, "UNDEF"}, {C, "C"}, {L, "L"}, {M, "M"}, {N, "N"}, {P, "P"}, {S, "S"}, {Z, "Z"},
            {Cc, "Cc"}, {Cf, "Cf"}, {Co, "Co"}, {Cs, "Cs"}, {Ll, "Ll"}, {Lm, "Lm"}, {Lo, "Lo"}, {Lt, "Lt"},
            {Lu, "Lu"}, {Mc, "Mc"}, {Me, "Me"}, {Mn, "Mn"}, {Nd, "Nd"}, {Nl, "Nl"}, {No, "No"}, {Pc, "Pc"},
            {Pd, "Pd"}, {Pe, "Pe"}, {Pf, "Pf"}, {Pi, "Pi"}, {Po, "Po"}, {Ps, "Ps"}, {Sc, "Sc"}, {Sk, "Sk"},
            {Sm, "Sm"}, {So, "So"}, {Zl, "Zl"}, {Zp, "Zp"}, {Zs, "Zs"},
        };
        const auto it = map.find(encoded & MASK);
        return it == map.end() ? "INVALID" : it->second;
    }

    static codepoint_categ from_index(int index) {
        static const std::array<codepoint_categ, 32> table = {
            UNDEF, Cc, Cf, Co, Cs, Ll, Lm, Lo, Lt, Lu, Mc, Me, Mn, Nd, Nl, No, Pc, Pd, Pe, Pf, Pi, Po, Ps, Sc, Sk, Sm, So, Zl, Zp, Zs, UNDEF, UNDEF
        };
        return (size_t)index < table.size() ? table[index] : table[0];
    }

    static codepoint_categ from_chars(const char categ, const char subcateg = '\0') {
        auto _subindex = [] (const char subcateg, const char subcategs[]) -> uint16_t {
            if (!subcateg) {
                return 0;
            }
            const char * p = strchr(subcategs, subcateg);
            GGML_ASSERT(p);
            return (uint16_t) (p - subcategs + 1);
        };
        switch(categ) {
            case 'C':  if(subcateg == 'n') return 0;  // undefined
                       return C | _subindex(subcateg, "cfos"   );
            case 'L':  return L | _subindex(subcateg, "lmotu"  );
            case 'M':  return M | _subindex(subcateg, "cen"    );
            case 'N':  return N | _subindex(subcateg, "dlo"    );
            case 'P':  return P | _subindex(subcateg, "cdefios");
            case 'S':  return S | _subindex(subcateg, "ckmo"   );
            case 'Z':  return Z | _subindex(subcateg, "lps"    );
            default:   GGML_ABORT("invalid category character");
        }
    }

    uint16_t encoded;
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8(uint32_t cp);
uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

codepoint_categ unicode_cpt_category(const uint32_t cp);
codepoint_categ unicode_cpt_category(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string & utf8);

uint32_t unicode_tolower(uint32_t cp);

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs);
