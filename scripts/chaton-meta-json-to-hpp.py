#!/usr/bin/env python3
# Convert chaton meta json file to equivalent c++ cpp format
# by Humans for All

import sys
import json


def kkv_str(j, tmpl, k1, k2, comma):
    print("\t\t{{ \"{}\", \"{}\" }}{}".format("{}-{}".format(k1,k2), repr(j[tmpl][k1][k2])[1:-1], comma))

def kv_str(j, tmpl, k1, comma):
    print("\t\t{{ \"{}\", \"{}\" }}{}".format(k1, repr(j[tmpl][k1])[1:-1], comma))

def kv_bool(j, tmpl, k1, comma):
    print("\t\t{{ \"{}\", {} }}{}".format(k1, repr(j[tmpl][k1]).lower(), comma))


fp=open(sys.argv[1])
j=json.load(fp)
print("//This is auto created/converted from chaton-meta-json file")
print("\n\n#include \"chaton.hpp\"\n\nChatTemplates gCT = {{")

for tmpl in j:
    print("\t{{ \"{}\", {{".format(tmpl))

    kkv_str(j, tmpl, "global", "begin", ",")
    kkv_str(j, tmpl, "global", "end", ",")

    kkv_str(j, tmpl, "system", "begin", ",")
    kkv_str(j, tmpl, "system", "prefix", ",")
    kkv_str(j, tmpl, "system", "suffix", ",")
    kkv_str(j, tmpl, "system", "end", ",")

    kkv_str(j, tmpl, "user", "begin", ",")
    kkv_str(j, tmpl, "user", "prefix", ",")
    kkv_str(j, tmpl, "user", "suffix", ",")
    kkv_str(j, tmpl, "user", "end", ",")

    kkv_str(j, tmpl, "assistant", "begin", ",")
    kkv_str(j, tmpl, "assistant", "prefix", ",")
    kkv_str(j, tmpl, "assistant", "suffix", ",")
    kkv_str(j, tmpl, "assistant", "end", ",")

    kv_str(j, tmpl, "reverse-prompt", ",")

    kv_bool(j, tmpl, "systemuser-system-has-suffix", ",")
    kv_bool(j, tmpl, "systemuser-system-has-end", ",")
    kv_bool(j, tmpl, "systemuser-1st-user-has-begin", ",")
    kv_bool(j, tmpl, "systemuser-1st-user-has-prefix", ",")

    print("\t}},")

print("}};")

