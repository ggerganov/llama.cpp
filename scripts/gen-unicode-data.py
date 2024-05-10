import regex


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
        print("const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_{} = {{".format(cat)) # noqa: NP100
    if mode == "map":
        print("const std::map<uint32_t, uint32_t> unicode_map_{} = {{".format(cat)) # noqa: NP100
    for i, values in enumerate(ranges):
        end = ",\n" if (i % 4 == 3 or i + 1 == len(ranges)) else ", "
        values = ["0x%08X" % value for value in values]
        print("{" + ", ".join(values) + "}", end=end) # noqa: NP100
    print("};") # noqa: NP100
    print("") # noqa: NP100


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
        map_lowercase.append((codepoint, lower))
    if codepoint != upper:
        map_uppercase.append((codepoint, upper))
print_cat("map", "lowercase", map_lowercase)
print_cat("map", "uppercase", map_uppercase)


# TODO: generate unicode_map_nfd
