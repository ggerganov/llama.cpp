import argparse
import os

import torch
from safetensors.torch import load_file
import numpy as np
from gguf import *

VISION = "clip.vision"

def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

def get_tensor_name(name: str) -> str:

    return name.replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("norm", "ln")


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-path", help=".pth model path", required=True)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)

# with proper
args = ap.parse_args()

model_path = args.model_path
model_name = os.path.basename(model_path).replace(".pth", "")
dir_model = os.path.dirname(model_path)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if args.use_f32:
    ftype = 0

# load the model
if model_path.endswith(".pth"):
    model = torch.load(model_path, map_location=torch.device('cpu'))
else:
    # model = GGUFReader(model_path)
    # tensors = model.tensors
    model = load_file(model_path)

# for t in tensors:
    # print(f"Name: {t.name}, data: {t.shape}, dtype: {t.tensor_type}")
# for name, data in model.items():
    # print(f"Name: {name}, data: {data.shape}, dtype: {data.dtype}")
# exit(0)

output_dir = args.output_dir if args.output_dir is not None else dir_model
# os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.basename(output_dir).replace("ggml_", "")
fname_out = os.path.join(output_dir, f"{model_name}-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch=model_name)

fout.add_file_type(ftype)
fout.add_name(model_name)

fout.add_description("Vision Transformer model")

with open(os.path.join(dir_model, "config.json"), "r", encoding="utf-8") as config_file:
    config = json.load(config_file)
    hparams = config["vision_config"]

fout.add_uint32("clip.vision.image_size", hparams["image_size"])
fout.add_uint32("clip.vision.patch_size", hparams["patch_size"])
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), hparams["hidden_size"])
fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), hparams["intermediate_size"])
# fout.add_uint32("clip.vision.projection_dim", hparams.get("projection_dim", config["projection_dim"]))
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), hparams["num_attention_heads"])
fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), hparams["layer_norm_eps"])
block_count = hparams["num_hidden_layers"]
fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)

with open(os.path.join(dir_model, "preprocessor_config.json"), "r", encoding="utf-8") as f:
    preprocessor_config = json.load(f)

image_mean = preprocessor_config["image_mean"]
image_std = preprocessor_config["image_std"]

fout.add_array("clip.vision.image_mean", image_mean)
fout.add_array("clip.vision.image_std", image_std)

for name, data in model.items():
    if name.find('language_model') != -1:
        continue
    name = get_tensor_name(name)
    data = data.float().numpy()
    # pw and dw conv ndim==4
    if (data.ndim == 2 or data.ndim == 4) and ftype == 1:
        data = data.astype(np.float16)
    # split in weight/bias into q,k,v
    if ".attn.qkv" in name:
        # [1024*3, 1024] -> 3*[1024, 1024]
        print(f"Splitting {name} with shape {data.shape}")
        if data.shape[0] == 1024*3:
            data = data.reshape(3, 1024, -1)
            qkv = [data[0].squeeze(), data[1].squeeze(), data[2].squeeze()]
        elif data.shape[0] == 1024*3:
            qkv = np.split(data, 3, axis=0)
        else:
            raise ValueError(f"Unknown shape {data.shape}")
        

        print(f"{name} shape {data.shape} split into {len(qkv)} shape: {qkv[0].shape}, {qkv[1].shape}, {qkv[2].shape}")
        fout.add_tensor(name.replace(".attn.qkv", ".attn.q"), qkv[0])
        fout.add_tensor(name.replace(".attn.qkv", ".attn.k"), qkv[1])
        fout.add_tensor(name.replace(".attn.qkv", ".attn.v"), qkv[2])
    else:
        fout.add_tensor(name, data)

fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()