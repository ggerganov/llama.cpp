import regex
import unicodedata


if False:

    # This code is equivalent to: cpt.to_bytes(4, "little"))
    def cpt_to_utf8_str(cpt):
        if cpt <= 0xFF:
            return bytes([cpt, 0, 0, 0])
        elif cpt <= 0xFFFF:
            return bytes([cpt & 0xFF, cpt >> 8, 0, 0])
        elif cpt <= 0xFFFFFF:
            return bytes([cpt & 0xFF, (cpt >> 8) & 0xFF, (cpt >> 16) & 0xFF, 0])
        else:
            return bytes([cpt & 0xFF, (cpt >> 8) & 0xFF, (cpt >> 16) & 0xFF, cpt >> 24])

    # This code is equivalent to: regex_expr_compiled.match(chr(codepoint))
    def is_match(codepoint, regex_expr):
        try:
            res = regex_expr.match(cpt_to_utf8_str(codepoint).decode('utf-32'))
            return res is not None
        except Exception:
            return False

    # Verify previous statements, using chr() and ord()
    for codepoint in range(0x110000):
        temp = cpt_to_utf8_str(codepoint)
        assert(temp == codepoint.to_bytes(4, "little"))
        try:
            char = temp.decode('utf-32')
            if codepoint == 0xFEFF:  # BOM
                assert(char == "")   # why?
                char = "\uFEFF"
        except UnicodeDecodeError:
            continue
        assert(char == chr(codepoint) )
        assert(ord(char) == codepoint )


def get_matches(regex_expr):
    regex_expr_compiled = regex.compile(regex_expr)
    unicode_ranges = []
    current_range = None

    for codepoint in range(0x110000):
        char = chr(codepoint)
        if regex_expr_compiled.match(char):
            if current_range is None:
                current_range = [codepoint, codepoint]
            else:
                current_range[1] = codepoint
        elif current_range is not None:
            unicode_ranges.append(tuple(current_range))
            current_range = None

    if current_range is not None:
        unicode_ranges.append(tuple(current_range))

    return unicode_ranges


def print_cat(mode, cat, ranges):
    if mode == "range":
        print("const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_{} = {{".format(cat))
    if mode == "range_value":
        print("const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> unicode_ranges_{} = {{".format(cat))
    if mode == "map":
        print("const std::map<uint32_t, uint32_t> unicode_map_{} = {{".format(cat))
    for i, values in enumerate(ranges):
        end = ",\n" if (i%4 == 3 or i+1 == len(ranges)) else ", "
        values = ["0x%08X"%value for value in values]
        print("{" + ", ".join(values) + "}", end=end)
    print("};")
    print("")


print_cat("range", "number",      get_matches(r'\p{N}'))
print_cat("range", "letter",      get_matches(r'\p{L}'))
print_cat("range", "separator",   get_matches(r'\p{Z}'))
print_cat("range", "accent_mark", get_matches(r'\p{M}'))
print_cat("range", "punctuation", get_matches(r'\p{P}'))
print_cat("range", "symbol",      get_matches(r'\p{S}'))
print_cat("range", "control",     get_matches(r'\p{C}'))

print_cat("range", "whitespace",  get_matches(r'\s'))


map_lowercase = []
map_uppercase = []
for codepoint in range(0x110000):
    char = chr(codepoint)
    lower = ord(char.lower()[0])
    upper = ord(char.upper()[0])
    if codepoint != lower:
        map_lowercase.append((codepoint,lower))
    if codepoint != upper:
        map_uppercase.append((codepoint,upper))
print_cat("map", "lowercase", map_lowercase)
print_cat("map", "uppercase", map_uppercase)


inv_map_nfd = {}
for codepoint in range(0x110000):
    char = chr(codepoint)
    norm = ord(unicodedata.normalize('NFD', char)[0])
    if codepoint != norm:
        a, b = inv_map_nfd.get(norm, (codepoint, codepoint))
        inv_map_nfd[norm] = (min(a, codepoint), max(b, codepoint))
nfd_ranges = [ (a, b, nfd) for nfd,(a,b) in inv_map_nfd.items() ]
nfd_ranges = list(sorted(nfd_ranges))
del inv_map_nfd
print_cat("range_value", "nfd", nfd_ranges)
