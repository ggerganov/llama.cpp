#!/usr/bin/env python3
# Convert chaton meta json file to c++ hpp file
# by Humans for All

import sys
import json


def kv(j, tmpl, k1, k2, comma):
    print("\t\t{{ \"{}\", \"{}\" }}{}".format("{}-{}".format(k1,k2), repr(j[tmpl][k1][k2])[1:-1], comma))


fp=open(sys.argv[1])
j=json.load(fp)
print("{")
for tmpl in j:
    print("\t{{ \"{}\", {{".format(tmpl))

    kv(j, tmpl, "global", "begin", ",")
    kv(j, tmpl, "global", "end", ",")

    kv(j, tmpl, "system", "begin", ",")
    kv(j, tmpl, "system", "prefix", ",")
    kv(j, tmpl, "system", "suffix", ",")
    kv(j, tmpl, "system", "end", ",")

    kv(j, tmpl, "user", "begin", ",")
    kv(j, tmpl, "user", "prefix", ",")
    kv(j, tmpl, "user", "suffix", ",")
    kv(j, tmpl, "user", "end", ",")

    kv(j, tmpl, "assistant", "begin", ",")
    kv(j, tmpl, "assistant", "prefix", ",")
    kv(j, tmpl, "assistant", "suffix", ",")
    kv(j, tmpl, "assistant", "end", ",")

    print("\t}},")
