import numpy
import json
import os
import sys
import time
import sh_common

if len(sys.argv) != 2:
    print("import_vgg7.py JSONPATH")
    print(" i.e. import_vgg7.py /home/you/Documents/External/waifu2x/models/vgg_7/art/scale2.0x_model.json")
    sys.exit(1)

try:
    os.mkdir("model-kipper")
except:
    pass

data_list = json.load(open(sys.argv[1], "rb"))

idx = 0
for i in range(7):
    layer = data_list[i]
    w = numpy.array(layer["weight"])
    w.reshape((-1, 3, 3)).transpose((0, 2, 1))
    b = numpy.array(layer["bias"])
    sh_common.save_param("kipper", idx, w)
    idx += 1
    sh_common.save_param("kipper", idx, b)
    idx += 1

