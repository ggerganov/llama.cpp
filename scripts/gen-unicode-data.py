import regex


def cpt_to_utf8_str(cpt):
    if cpt <= 0xFF:
        return bytes([cpt, 0, 0, 0])
    elif cpt <= 0xFFFF:
        return bytes([cpt & 0xFF, cpt >> 8, 0, 0])
    elif cpt <= 0xFFFFFF:
        return bytes([cpt & 0xFF, (cpt >> 8) & 0xFF, (cpt >> 16) & 0xFF, 0])
    else:
        return bytes([cpt & 0xFF, (cpt >> 8) & 0xFF, (cpt >> 16) & 0xFF, cpt >> 24])


def is_match(codepoint, regex_expr):
    try:
        res = regex.match(regex_expr, cpt_to_utf8_str(codepoint).decode('utf-32'))
        return res is not None
    except Exception:
        return False


def get_matches(regex_expr):
    unicode_ranges = []
    current_range = None

    for codepoint in range(0x110000):
        if is_match(codepoint, regex_expr):
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


def print_cat(cat, ranges):
    print("const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_{} = {{".format(cat)) # noqa: NP100
    cnt = 0
    for start, end in ranges:
        if cnt % 4 != 0:
            print(" ", end="") # noqa: NP100
        print("{{0x{:08X}, 0x{:08X}}},".format(start, end), end="") # noqa: NP100
        if cnt % 4 == 3:
            print("") # noqa: NP100
        cnt += 1

    if cnt % 4 != 0:
        print("") # noqa: NP100
    print("};") # noqa: NP100
    print("") # noqa: NP100


print_cat("number",      get_matches(r'\p{N}'))
print_cat("letter",      get_matches(r'\p{L}'))
print_cat("whitespace",  get_matches(r'\p{Z}'))
print_cat("accent_mark", get_matches(r'\p{M}'))
print_cat("punctuation", get_matches(r'\p{P}'))
print_cat("symbol",      get_matches(r'\p{S}'))
print_cat("control",     get_matches(r'\p{C}'))
