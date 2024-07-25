from __future__ import annotations

import array
import unicodedata
import requests


MAX_CODEPOINTS = 0x110000

UNICODE_DATA_URL = "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt"


# see https://www.unicode.org/L2/L1999/UnicodeData.html
def unicode_data_iter():
    res = requests.get(UNICODE_DATA_URL)
    res.raise_for_status()
    data = res.content.decode()

    prev = []

    for line in data.splitlines():
        # ej: 0000;<control>;Cc;0;BN;;;;;N;NULL;;;;
        line = line.split(";")

        cpt = int(line[0], base=16)
        assert cpt < MAX_CODEPOINTS

        cpt_lower = int(line[-2] or "0", base=16)
        assert cpt_lower < MAX_CODEPOINTS

        cpt_upper = int(line[-3] or "0", base=16)
        assert cpt_upper < MAX_CODEPOINTS

        categ = line[2].strip()
        assert len(categ) == 2

        bidir = line[4].strip()
        assert len(categ) == 2

        name = line[1]
        if name.endswith(", First>"):
            prev = (cpt, cpt_lower, cpt_upper, categ, bidir)
            continue
        if name.endswith(", Last>"):
            assert prev[1:] == (0, 0, categ, bidir)
            for c in range(prev[0], cpt):
                yield (c, cpt_lower, cpt_upper, categ, bidir)

        yield (cpt, cpt_lower, cpt_upper, categ, bidir)


# see codepoint_categ::from_index() in unicode.h
UNICODE_CATEGORY_TO_INDEX = {
    "Cn":  0,  # \p{Cn} Undefined
    "Cc":  1,  # \p{Cc} Control
    "Cf":  2,  # \p{Cf} Format
    "Co":  3,  # \p{Co} Private Use
    "Cs":  4,  # \p{Cs} Surrrogate
    "Ll":  5,  # \p{Ll} Lowercase Letter
    "Lm":  6,  # \p{Lm} Modifier Letter
    "Lo":  7,  # \p{Lo} Other Letter
    "Lt":  8,  # \p{Lt} Titlecase Letter
    "Lu":  9,  # \p{Lu} Uppercase Letter
    "Mc": 10,  # \p{Mc} Spacing Mark
    "Me": 11,  # \p{Me} Enclosing Mark
    "Mn": 12,  # \p{Mn} Nonspacing Mark
    "Nd": 13,  # \p{Nd} Decimal Number
    "Nl": 14,  # \p{Nl} Letter Number
    "No": 15,  # \p{No} Other Number
    "Pc": 16,  # \p{Pc} Connector Punctuation
    "Pd": 17,  # \p{Pd} Dash Punctuation
    "Pe": 18,  # \p{Pe} Close Punctuation
    "Pf": 19,  # \p{Pf} Final Punctuation
    "Pi": 20,  # \p{Pi} Initial Punctuation
    "Po": 21,  # \p{Po} Other Punctuation
    "Ps": 22,  # \p{Ps} Open Punctuation
    "Sc": 23,  # \p{Sc} Currency Symbol
    "Sk": 24,  # \p{Sk} Modifier Symbol
    "Sm": 25,  # \p{Sm} Math Symbol
    "So": 26,  # \p{So} Other Symbol
    "Zl": 27,  # \p{Zl} Line Separator
    "Zp": 28,  # \p{Zp} Paragraph Separator
    "Zs": 29,  # \p{Zs} Space Separator
}


codepoint_categs = array.array('B', [0]) * MAX_CODEPOINTS  # Undefined
table_whitespace = []
table_lowercase = []
table_uppercase = []
table_nfd = []

for (cpt, cpt_lower, cpt_upper, categ, bidir) in unicode_data_iter():
    # convert codepoint to unicode character
    char = chr(cpt)

    # codepoint category flags
    codepoint_categs[cpt] = UNICODE_CATEGORY_TO_INDEX[categ]

    # lowercase conversion
    if cpt_lower:
        table_lowercase.append((cpt, cpt_lower))

    # uppercase conversion
    if cpt_upper:
        table_uppercase.append((cpt, cpt_upper))

    # NFD normalization
    norm = ord(unicodedata.normalize('NFD', char)[0])
    if cpt != norm:
        table_nfd.append((cpt, norm))


# whitespaces, see "<White_Space>" https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
table_whitespace.extend(range(0x0009, 0x000D + 1))
table_whitespace.extend(range(0x2000, 0x200A + 1))
table_whitespace.extend([0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000])


# sort by codepoint
table_whitespace.sort()
table_lowercase.sort()
table_uppercase.sort()
table_nfd.sort()


# run length encoding, see unicode_cpt_category() in unicode.cpp
assert (max(UNICODE_CATEGORY_TO_INDEX.values()) < 32)
codepoint_categs_runs = [codepoint_categs[0]]  # 5 bits categ + 11 bits length
for cpt, categ in enumerate(codepoint_categs[1:], 1):
    prev = codepoint_categs_runs[-1]
    if prev <= (0xFFFF - 32) and (prev & 31) == categ:
        codepoint_categs_runs[-1] += 32  # increment run length
    else:
        codepoint_categs_runs.append(categ)  # new run value
    assert (codepoint_categs_runs[-1] < 0xFFFF)
assert (MAX_CODEPOINTS == sum((rle >> 5) + 1 for rle in codepoint_categs_runs))


# group ranges with same nfd
ranges_nfd: list[tuple[int, int, int]] = [(0, 0, 0)]  # start, last, nfd
for codepoint, norm in table_nfd:
    start = ranges_nfd[-1][0]
    if ranges_nfd[-1] != (start, codepoint - 1, norm):
        ranges_nfd.append(None)  # type: ignore[arg-type]  # dummy, will be replaced below
        start = codepoint
    ranges_nfd[-1] = (start, codepoint, norm)


# Generate 'unicode-data.cpp':
#   python ./scripts//gen-unicode-data.py > ./src/unicode-data.cpp

def out(line=""):
    print(line, end='\n')  # noqa


out("""\
// generated with scripts/gen-unicode-data.py

#include "unicode-data.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
""")

out("const std::vector<uint16_t> unicode_rle_codepoints_categs = {  // run length encoding, 5 bits categ + 11 bits length")
for rle in codepoint_categs_runs:
    out("0x%04X," % rle)
out("};\n")

out("const std::vector<uint32_t> unicode_vec_whitespace = {")
for codepoint in table_whitespace:
    out("0x%06X," % codepoint)
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase = {")
for tuple_lw in table_lowercase:
    out("{0x%06X, 0x%06X}," % tuple_lw)
out("};\n")

out("const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase = {")
for tuple_up in table_uppercase:
    out("{0x%06X, 0x%06X}," % tuple_up)
out("};\n")

out("const std::vector<range_nfd> unicode_ranges_nfd = {  // start, last, nfd")
for triple in ranges_nfd:
    out("{0x%06X, 0x%06X, 0x%06X}," % triple)
out("};\n")
