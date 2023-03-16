#!/usr/bin/env python3

import os
import sys

if not (len(sys.argv) == 2 and sys.argv[1] in ["7B", "13B", "30B", "65B"]):
    print(f"\nUsage: {sys.argv[0]} 7B|13B|30B|65B [--remove-f16]\n")
    sys.exit(1)

for i in os.listdir(f"models/{sys.argv[1]}"):
    if i.endswith("ggml-model-f16.bin.1"):
        os.system(f"quantize.exe {os.path.join('models', sys.argv[1], i)} {os.path.join('models', sys.argv[1], i.replace('f16', 'q4_0'))} 2")
        print(f"quantize.exe {os.path.join('models', sys.argv[1], i)} {os.path.join('models', sys.argv[1], i.replace('f16', 'q4_1'))} 2")
        if len(sys.argv) == 3 and sys.argv[2] == "--remove-f16":
            os.remove(os.path.join('models', sys.argv[1], i))