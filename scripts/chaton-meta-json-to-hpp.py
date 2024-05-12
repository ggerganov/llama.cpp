#!/usr/bin/env python3
# Convert chaton meta json file to c++ hpp file
# by Humans for All

import sys
import json


def kv(j, tmpl, k1, k2, comma):
  print("\t\t{{ \"{}\", \"{}\" }}{}".format("{}-{}".format(k1,k2), j[tmpl][k1][k2], comma))


fp=open(sys.argv[1])
j=json.load(fp)
print("{")
for tmpl in j:
  print("\t{{ \"{}\", {{".format(tmpl))
  kv(j, tmpl, "global", "begin", ",")
  kv(j, tmpl, "global", "end", ",")
  print("\t}},")
 
