#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <map>

struct codepoint_categ {
    enum _category : uint16_t {
        UNDEF = 0,   // \p{Cn} Undefined
        C = 1 << 0,  // \p{C}  Control
        L = 1 << 1,  // \p{L}  Letter
        M = 1 << 2,  // \p{M}  Mark
        N = 1 << 3,  // \p{N}  Number
        P = 1 << 4,  // \p{P}  Punctuation
        S = 1 << 5,  // \p{S}  Symbol
        Z = 1 << 6,  // \p{Z}  Separator
        MASK = (1 << 7) - 1  // 7 bits
    };

    enum _subcategory : uint16_t {
        Cc = C | (1 << 7),  // \p{Cc} Control
        Cf = C | (2 << 7),  // \p{Cf} Format
        Co = C | (3 << 7),  // \p{Co} Private Use
        Cs = C | (4 << 7),  // \p{Cs} Surrrogate
        Ll = L | (1 << 7),  // \p{Ll} Lowercase Letter
        Lm = L | (2 << 7),  // \p{Lm} Modifier Letter
        Lo = L | (3 << 7),  // \p{Lo} Other Letter
        Lt = L | (4 << 7),  // \p{Lt} Titlecase Letter
        Lu = L | (5 << 7),  // \p{Lu} Uppercase Letter
        Mc = M | (1 << 7),  // \p{Mc} Spacing Mark
        Me = M | (2 << 7),  // \p{Me} Enclosing Mark
        Mn = M | (3 << 7),  // \p{Mn} Nonspacing Mark
        Nd = N | (1 << 7),  // \p{Nd} Decimal Number
        Nl = N | (2 << 7),  // \p{Nl} Letter Number
        No = N | (3 << 7),  // \p{No} Other Number
        Pc = P | (1 << 7),  // \p{Pc} Connector Punctuation
        Pd = P | (2 << 7),  // \p{Pd} Dash Punctuation
        Pe = P | (3 << 7),  // \p{Pe} Close Punctuation
        Pf = P | (4 << 7),  // \p{Pf} Final Punctuation
        Pi = P | (5 << 7),  // \p{Pi} Initial Punctuation
        Po = P | (6 << 7),  // \p{Po} Other Punctuation
        Ps = P | (7 << 7),  // \p{Ps} Open Punctuation
        Sc = S | (1 << 7),  // \p{Sc} Currency Symbol
        Sk = S | (2 << 7),  // \p{Sk} Modifier Symbol
        Sm = S | (3 << 7),  // \p{Sm} Math Symbol
        So = S | (4 << 7),  // \p{So} Other Symbol
        Zl = Z | (1 << 7),  // \p{Zl} Line Separator
        Zp = Z | (2 << 7),  // \p{Zp} Paragraph Separator
        Zs = Z | (3 << 7),  // \p{Zs} Space Separator
        SUBMASK = (1 << 10) - 1  // 7+3 bits
    };

    enum _flags : uint16_t {
        WHITESPACE = (1 << 10),  // regex: \s
        LOWERCASE  = (1 << 11),
        UPPERCASE  = (1 << 12),
        //Norm NFD/NFC  = ...,
    };

    inline codepoint_categ(const uint16_t categ=0) : encoded{categ} {}

    static codepoint_categ from_index(int index) {
        static const std::array<codepoint_categ, 32> table = {
            UNDEF, Cc, Cf, Co, Cs, Ll, Lm, Lo, Lt, Lu, Mc, Me, Mn, Nd, Nl, No, Pc, Pd, Pe, Pf, Pi, Po, Ps, Sc, Sk, Sm, So, Zl, Zp, Zs, UNDEF, UNDEF
        };
        return (size_t)index < table.size() ? table[index] : table[0];
    }

    inline void set_flag(_flags flags, bool value = true) {
        flags = (_flags) (flags & ~SUBMASK);  // ignore category bits
        encoded = value ? (encoded | flags) : (encoded & ~flags);
    }

    inline uint8_t get_category() const { return encoded & MASK; }
    inline uint8_t get_subcategory() const { return encoded & SUBMASK; }

    inline bool is_undefined() const { return !encoded; }
    inline bool is_defined() const { return encoded; }

    inline auto is_whitespace() const { return encoded & WHITESPACE; }
    inline auto is_lowercase()  const { return encoded & LOWERCASE; }
    inline auto is_uppercase()  const { return encoded & UPPERCASE; }

    inline auto is_C() const { return encoded & C; }
    inline auto is_L() const { return encoded & L; }
    inline auto is_M() const { return encoded & M; }
    inline auto is_N() const { return encoded & N; }
    inline auto is_P() const { return encoded & P; }
    inline auto is_S() const { return encoded & S; }
    inline auto is_Z() const { return encoded & Z; }

    inline auto is_Cc() const { return (encoded & SUBMASK) == Cc; }
    inline auto is_Cf() const { return (encoded & SUBMASK) == Cf; }
    inline auto is_Co() const { return (encoded & SUBMASK) == Co; }
    inline auto is_Cs() const { return (encoded & SUBMASK) == Cs; }
    inline auto is_Ll() const { return (encoded & SUBMASK) == Ll; }
    inline auto is_Lm() const { return (encoded & SUBMASK) == Lm; }
    inline auto is_Lo() const { return (encoded & SUBMASK) == Lo; }
    inline auto is_Lt() const { return (encoded & SUBMASK) == Lt; }
    inline auto is_Lu() const { return (encoded & SUBMASK) == Lu; }
    inline auto is_Mc() const { return (encoded & SUBMASK) == Mc; }
    inline auto is_Me() const { return (encoded & SUBMASK) == Me; }
    inline auto is_Mn() const { return (encoded & SUBMASK) == Mn; }
    inline auto is_Nd() const { return (encoded & SUBMASK) == Nd; }
    inline auto is_Nl() const { return (encoded & SUBMASK) == Nl; }
    inline auto is_No() const { return (encoded & SUBMASK) == No; }
    inline auto is_Pc() const { return (encoded & SUBMASK) == Pc; }
    inline auto is_Pd() const { return (encoded & SUBMASK) == Pd; }
    inline auto is_Pe() const { return (encoded & SUBMASK) == Pe; }
    inline auto is_Pf() const { return (encoded & SUBMASK) == Pf; }
    inline auto is_Pi() const { return (encoded & SUBMASK) == Pi; }
    inline auto is_Po() const { return (encoded & SUBMASK) == Po; }
    inline auto is_Ps() const { return (encoded & SUBMASK) == Ps; }
    inline auto is_Sc() const { return (encoded & SUBMASK) == Sc; }
    inline auto is_Sk() const { return (encoded & SUBMASK) == Sk; }
    inline auto is_Sm() const { return (encoded & SUBMASK) == Sm; }
    inline auto is_So() const { return (encoded & SUBMASK) == So; }
    inline auto is_Zl() const { return (encoded & SUBMASK) == Zl; }
    inline auto is_Zp() const { return (encoded & SUBMASK) == Zp; }
    inline auto is_Zs() const { return (encoded & SUBMASK) == Zs; }

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
        const auto it = map.find(encoded & SUBMASK);
        return it == map.end() ? "INVALID" : it->second;
    }

    uint16_t encoded;
};


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
