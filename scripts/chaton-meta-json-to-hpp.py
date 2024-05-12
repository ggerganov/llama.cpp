#!/usr/bin/env python3
# Convert chaton meta json file to c++ hpp file
# by Humans for All

import sys
import json

fp=open(sys.argv[1])
j=json.load(fp)
print("{")
for tmpl in j:
  print("\t{{ \"{}\", ".format(tmpl))
 
