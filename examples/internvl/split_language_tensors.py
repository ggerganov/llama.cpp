import argparse
import os

import torch
from safetensors.torch import load_file, save_file
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-path", help=".pth model path", required=True)
ap.add_argument("-o", "--output-path", help="Path to save language model. Default is the original model directory", default=None)

args = ap.parse_args()
model_path = args.model_path
model = load_file(model_path)
dir_model = os.path.dirname(model_path)
output_path = args.output_path if args.output_path is not None else os.path.join(dir_model, "language_model.safetensors")

# print(os.path.getsize("language_model.safetensors"))

language_tensors = {}
for name, data in model.items():
    print(f"Name: {name}, data: {data.shape}, dtype: {data.dtype}")
    if name.find("language_model.") != -1:
        language_tensors[name.replace("language_model.", "")] = data

save_file(language_tensors, output_path)

