import argparse
import os
import json
import re

import torch
import numpy as np
from gguf import *
# from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

VISION = "siglip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def should_skip_tensor(name: str, has_text: bool, has_vision: bool, has_omni_vlm: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    # if name.startswith("vision_model.post_layernorm") or name.startswith("vision_model.head"):
    #     return True

    if name.startswith("v") and not has_vision:
        return True

    if name.startswith("t") and not has_text:
        return True

    return False


def get_tensor_name(name: str) -> str:
    if "projection" in name:
        return name
    if "multi_modal_projector" in name:
        name = name.replace("multi_modal_projector", "mm")
        name = re.sub(r'mm\.mlp\.mlp', 'mm.model.mlp', name, count=1)
        name = re.sub(r'mm\.peg\.peg', 'mm.model.peg', name, count=1)
        return name

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model-dir", help="Path to model directory cloned from HF Hub", required=True)
ap.add_argument("-p", "--processor-dir", help="Path to vlm-processor directory cloned from HF Hub", required=True)
ap.add_argument("-o", "--output-dir", help="Directory to save GGUF files. Default is the original model directory", default=None)
ap.add_argument("--use-f32", action="store_true", default=False, help="Use f32 instead of f16")
# TODO: whether update this info?
# default_image_mean = [0.48145466, 0.4578275, 0.40821073]
# default_image_std = [0.26862954, 0.26130258, 0.27577711]
default_image_mean = [0.5, 0.5, 0.5]
default_image_std = [0.5, 0.5, 0.5]

# with proper
args = ap.parse_args()

if args.use_f32:
    print("WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")

# output in the same directory as the model if output_dir is None
dir_model = args.model_dir
dir_processor = args.processor_dir

with open(dir_processor + "/preprocessor_config.json", "r", encoding="utf-8") as f:
    hparams = json.load(f)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if args.use_f32:
    ftype = 0

has_omni_vlm_projector = True
fname_middle = "mmproj-"
output_dir = args.output_dir if args.output_dir is not None else dir_model
os.makedirs(output_dir, exist_ok=True)

fname_out = os.path.join(output_dir, f"{fname_middle}omni-vlm-{ftype_str[ftype]}.gguf")
fout = GGUFWriter(path=fname_out, arch="siglip")

fout.add_bool("siglip.has_omni_vlm_projector", has_omni_vlm_projector)
fout.add_file_type(ftype)
fout.add_name("omni-vlm")
fout.add_description("image encoder for omni-vlm")


fout.add_uint32("siglip.vision.image_size", 384)
fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), 1152)
fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), 16) #TODO: to be confirmed
fout.add_uint32("siglip.vision.patch_size", 14)
# block_count = (27 - 1)
block_count = 27
fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)

fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 4304)
fout.add_uint32("siglip.vision.projection_dim", 4096)
    # fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), v_hparams["layer_norm_eps"])
    # if "image_grid_pinpoints" in v_hparams:
    #     # flatten it
    #     image_grid_pinpoints = []
    #     for pinpoint in v_hparams["image_grid_pinpoints"]:
    #         for p in pinpoint:
    #             image_grid_pinpoints.append(p)
#     fout.add_array("clip.vision.image_grid_pinpoints", image_grid_pinpoints)
#     if "image_crop_resolution" in v_hparams:
#         fout.add_uint32("clip.vision.image_crop_resolution", v_hparams["image_crop_resolution"])
#     if "image_aspect_ratio" in v_hparams:
#         fout.add_string("clip.vision.image_aspect_ratio", v_hparams["image_aspect_ratio"])
#     if "image_split_resolution" in v_hparams:
#         fout.add_uint32("clip.vision.image_split_resolution", v_hparams["image_split_resolution"])
#     if "mm_patch_merge_type" in v_hparams:
#         fout.add_string("clip.vision.mm_patch_merge_type", v_hparams["mm_patch_merge_type"])
#     if "mm_projector_type" in v_hparams:
#         fout.add_string("clip.vision.mm_projector_type", v_hparams["mm_projector_type"])
#
#
#     if processor is not None:
#         image_mean = processor.image_processor.image_mean if args.image_mean is None or args.image_mean == default_image_mean else args.image_mean  # pyright: ignore[reportAttributeAccessIssue]
#         image_std = processor.image_processor.image_std if args.image_std is None or args.image_std == default_image_std else args.image_std  # pyright: ignore[reportAttributeAccessIssue]
#     else:
#         image_mean = args.image_mean if args.image_mean is not None else default_image_mean
#         image_std = args.image_std if args.image_std is not None else default_image_std
#     fout.add_array("clip.vision.image_mean", image_mean)
#     fout.add_array("clip.vision.image_std", image_std)
#
fout.add_array("siglip.vision.image_mean", default_image_mean)
fout.add_array("siglip.vision.image_std", default_image_std)

# use_gelu = v_hparams["hidden_act"] == "gelu"
# fout.add_bool("clip.use_gelu", use_gelu)

model = torch.load(os.path.join(dir_model, "omni_vlm.clip"), map_location='cpu')
    # model.vision_model.encoder.layers.pop(-1)
projector = torch.load(os.path.join(dir_model, "omni_vlm.projector"), map_location='cpu')
for name, data in projector.items():
    name = get_tensor_name(name)
    # pw and dw conv ndim==4
    if data.ndim == 2 or data.ndim == 4:
        data = data.squeeze().cpu().numpy().astype(np.float16)
    else:
        data = data.squeeze().cpu().numpy().astype(np.float32)

    fout.add_tensor(name, data)

print("Projector tensors added\n")


# state_dict = model.state_dict()
state_dict = dict(model)
for name, data in state_dict.items():
    if should_skip_tensor(name, False, True, True):
        # we don't need this
        print(f"skipping parameter: {name}")
        continue

    # if name.startswith(f"vision_model.encoder.layers.{block_count}"):
    #     continue

    name = get_tensor_name(name)
    # data = data.astype(np.float16)
    # print(data)
    data = data.squeeze().float().numpy()

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if n_dims == 4:
        print(f"tensor {name} is always saved in f16")
        data = data.astype(np.float16)
        ftype_cur = 1
    elif ftype == 1:
        if name[-7:] == ".weight" and n_dims == 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    print(f"{name} - {ftype_str[ftype_cur]} - shape = {data.shape}")
    fout.add_tensor(name, data)


fout.write_header_to_file()
fout.write_kv_data_to_file()
fout.write_tensors_to_file()
fout.close()

print("Done. Output file: " + fname_out)
