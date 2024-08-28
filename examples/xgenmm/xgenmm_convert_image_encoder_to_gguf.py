import os
import re
import torch
import argparse
import json
import numpy as np
import time

from gguf import *
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer, SiglipVisionConfig

TEXT = "clip.text"
VISION = "clip.vision"

def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)

def should_skip_tensor(name: str, has_text: bool, has_vision: bool, has_minicpmv: bool) -> bool:
    if name in (
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ):
        return True

    if has_minicpmv and name in ["visual_projection.weight"]:
        return True

    if name.startswith("v") and not has_vision:
        return True

    if name.startswith("t") and not has_text:
        return True

    return False


def get_tensor_name(name: str) -> str:
    if "projection" in name:
        return name
    if "mm_projector" in name:
        name = name.replace("model.mm_projector", "mm")
        name = re.sub(r'mm\.mlp\.mlp', 'mm.model.mlp', name, count=1)
        name = re.sub(r'mm\.peg\.peg', 'mm.model.peg', name, count=1)
        return name

    return name.replace("text_model", "t").replace("vision_model", "v").replace("encoder.layers", "blk").replace("embeddings.", "").replace("_proj", "").replace("self_attn.", "attn_").replace("layer_norm", "ln").replace("layernorm", "ln").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("embedding", "embd").replace("final", "post").replace("layrnorm", "ln")


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

######################################### 
#### belows are added for xgenmm
########################################

def _replace_name_vit(s,v):
    s = "vision_model." + s
    if re.match("vision_model.embeddings.position_embedding", s):
        v = v.unsqueeze(0)
        return {s: v}
    return {s: v}

def _replace_attn_layer(key, value):
    # Check for the special case first
    if re.match(r'layers\.(\d+)\.0\.to_kv\.weight', key):
        idx = re.search(r'layers\.(\d+)\.0\.to_kv\.weight', key).group(1)
        KVweight = value.chunk(2, dim=0)
        return {f'blk.{idx}.attn.to_k.weight': KVweight[0],
                f'blk.{idx}.attn.to_v.weight': KVweight[1]
                }
    
    # Apply general replacements for other patterns
    # Define the replacement patterns
    patterns = [
        (r'layers\.(\d+)\.0\.norm_media\.(weight|bias)', r'blk.\1.attn.norm_media.\2'),
        (r'layers\.(\d+)\.0\.norm_latents\.(weight|bias)', r'blk.\1.attn.norm_latents.\2'),
        (r'layers\.(\d+)\.0\.to_q\.(weight)', r'blk.\1.attn.to_q.\2'),
        (r'layers\.(\d+)\.0\.to_out\.(weight)', r'blk.\1.attn.to_out.\2'),
        (r'layers\.(\d+)\.1\.0\.(weight|bias)', r'blk.\1.ffn.ln.\2'),
        (r'layers\.(\d+)\.1\.1\.weight', r'blk.\1.ffn.linear_up.weight'),
        (r'layers\.(\d+)\.1\.3\.weight', r'blk.\1.ffn.linear_down.weight'),
    ]
    for pattern, replacement in patterns:
        key = re.sub(pattern, replacement, key)
    
    return {key: value}

def replace_tensor_name_xgenmm_projector(ckpt):
    identifier = 'perceiver_resampler.'
    new_state_dict = {}
    for k, v in ckpt.items():
        # handel the layer
        if 'layers' in k:
            new_kvs = _replace_attn_layer(k, v)
            for new_k, new_v in new_kvs.items():
                new_state_dict[identifier+new_k] = new_v
        elif k == 'norm.weight':
            new_k = 'ln.weight'
            new_state_dict[identifier+new_k] = v
        elif k == 'norm.bias':
            new_k = 'ln.bias'
            new_state_dict[identifier+new_k] = v  
        else:
            new_state_dict[identifier+k] = v
    return new_state_dict  

class print_time():
    def __init__(self, task):
        self.task = task
        
    def __enter__(self):
        print(f"🟡 {self.task}")
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'🟢 time used: [{time.time() - self.t:.03f}] secs')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surgery_dir", type=str, default='/export/share/yutong/xgenmm/llamacpp_wd')
    parser.add_argument('--version', type=str, default='siglip_kosmos_phi3_4k_instruct', help='help identify the version of the saved ckpt')
    # options kept from llama.cpp projects
    parser.add_argument("--use_f32", action="store_true", default=False, help="Use f32 instead of f16")
    parser.add_argument("--text_only", action="store_true", required=False,
                help="Save a text-only model. It can't be used to encode images")
    parser.add_argument("--vision_only", action="store_true", required=False,
                help="Save a vision-only model. It can't be used to encode texts")
    parser.add_argument("--xgenmm_projector", help="Path to xgenmm projector file. If specified, save an image encoder for XgenMM models.")
    parser.add_argument("--xgenmm_vit", help="Path to vit file.")
    parser.add_argument("--output_dirname", default="gguf",help="Output directory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.text_only and args.vision_only:
        print("--text-only and --image-only arguments cannot be specified at the same time.")
        exit(1)

    if args.use_f32:
        print("🟡 WARNING: Weights for the convolution op is always saved in f16, as the convolution op in GGML does not support 32-bit kernel weights yet.")
        
    # possible data types
    #   ftype == 0 -> float32
    #   ftype == 1 -> float16
    #
    # map from ftype to string
    ftype_str = ["f32", "f16"]

    ftype = 1
    if args.use_f32:
        ftype = 0        
    
    ckpt_dir = f"{args.surgery_dir}/{args.version}"
    if args.xgenmm_projector is None:
        args.xgenmm_projector = f"{ckpt_dir}/xgenmm.projector"
    if args.xgenmm_vit is None:
        args.xgenmm_vit = f"{ckpt_dir}/vision_encoder/xgenmm.vision_encoder"
    output_dir = f"{ckpt_dir}/{args.output_dirname}"
    
    
    vision_encoder_config_path = f"{args.surgery_dir}/{args.version}/vision_encoder/config.json"
    with open(vision_encoder_config_path, 'r') as f:
        vision_config = json.load(f)
    
    fname_middle = None
    has_text_encoder = True
    has_vision_encoder = True
    has_xgenmm_projector = False
    if args.text_only:
        fname_middle = "text-"
        has_vision_encoder = False
    elif args.xgenmm_projector is not None:
        fname_middle = "mmproj-"
        has_text_encoder = False
        has_xgenmm_projector = True
    elif args.vision_only:
        fname_middle = "vision-"
        has_text_encoder = False
    else:
        fname_middle = ""


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fname_out = os.path.join(output_dir, f"{fname_middle}model-{ftype_str[ftype]}.gguf")
    
    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_bool("clip.has_text_encoder", has_text_encoder)
    fout.add_bool("clip.has_vision_encoder", has_vision_encoder)
    fout.add_bool("clip.has_xgenmm_projector", has_xgenmm_projector)
    fout.add_file_type(ftype)
    
    if args.text_only:
        fout.add_description("text-only CLIP model")
    elif args.vision_only and not has_xgenmm_projector:
        fout.add_description("vision-only CLIP model")
    elif has_xgenmm_projector:
        fout.add_description("image encoder for XgenMM model")
        # add projector type
        fout.add_string("clip.projector_type", "PerceiverResampler")
    else:
        fout.add_description("two-tower CLIP model")

    if has_vision_encoder:
        """
        In siglip config, we have following keys
            used: "image_size", "patch_size", "hidden_size", "intermediate_size"
                        "num_attention_heads", "layer_norm_eps", "num_hidden_layers", "hidden_act"
            unused: "attention_dropout", "model_type", "num_channels"
        """
        with print_time("add vit configs to gguf"):
            fout.add_uint32("clip.vision.image_size", vision_config["image_size"])
            fout.add_uint32("clip.vision.patch_size", vision_config["patch_size"])
            fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vision_config["hidden_size"])
            fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), vision_config["intermediate_size"])
            # TODO: need to check the value of projection_dim; follow minicpmv to set it as 0
            fout.add_uint32("clip.vision.projection_dim", 0)
            fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vision_config["num_attention_heads"])
            fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), vision_config["layer_norm_eps"])
            # TODO: chekck this as it might causes bugs
            # orginial llaval implementation:
            # block_count = vision_config["num_hidden_layers"] - 1 if has_xgenmm_projector else vision_config["num_hidden_layers"]
            # we are different from llama1.6, which used the second to the last layer's hidden states as the image features.
            block_count = vision_config["num_hidden_layers"] 
            fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), block_count)
            # xgenmm use anyres with grids configuration
            # 1*2, 2*1, 2*2, 3*1, 1*3, the same as the llava1.6, we just hard code it here
            # the base resolution is 384
            image_grid_pinpoints = [384, 768, 768, 384, 768, 768, 1152, 384, 384, 1152]
            fout.add_array("clip.vision.image_grid_pinpoints", image_grid_pinpoints)
            
            
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]
            fout.add_array("clip.vision.image_mean", image_mean)
            fout.add_array("clip.vision.image_std", image_std)
            
            # vision_config["hidden_act"] is gelu_pytorch_tanh
            # ggml implements gelu_with_tanh approximation
            use_gelu = "gelu" in vision_config["hidden_act"].lower()
            fout.add_bool("clip.use_gelu", use_gelu)
            fout.add_string("clip.vision.mm_patch_merge_type", 'spatial_unpad')
            print("hard coded mm_patch_merge_type as spatial_unpad")
    
    # for VIT model
    with print_time("Loading vision encoder and converting to gguf"):
        vision_encoder_config = SiglipVisionConfig(**vision_config)
        vision_encoder = SiglipVisionTransformer(vision_encoder_config)
        vision_encoder_ckpt = torch.load(f'{ckpt_dir}/vision_encoder/xgenmm.vision_encoder')
        vision_encoder.load_state_dict(vision_encoder_ckpt)
        state_dict = vision_encoder.state_dict()
        new_state_dict = {}
        for k_, v_ in state_dict.items():
            kvs = _replace_name_vit(k_, v_)
            for nk, nv in kvs.items():
                # split in_proj_weight to q_proj_weight, k_proj_weight, v_proj_weight
                if nk == "vision_model.head.attention.in_proj_weight":
                    dim = int(nv.shape[0] / 3)
                    nk_1 = "vision_model.head.attention.q_proj_weight"
                    nv_1 = nv[:dim, :]
                    nk_2 = "vision_model.head.attention.k_proj_weight"
                    nv_2 = nv[dim:2*dim, :]
                    nk_3 = "vision_model.head.attention.v_proj_weight"
                    nv_3 = nv[2*dim:, :]
                    new_state_dict[nk_1] = nv_1
                    new_state_dict[nk_2] = nv_2
                    new_state_dict[nk_3] = nv_3
                # split in_proj_bias to q_proj_bias, k_proj_bias, v_proj_bias
                elif nk == "vision_model.head.attention.in_proj_bias":
                    dim = int(nv.shape[0] / 3)
                    nk_1 = "vision_model.head.attention.q_proj_bias"
                    nv_1 = nv[:dim]
                    nk_2 = "vision_model.head.attention.k_proj_bias"
                    nv_2 = nv[dim:2*dim]
                    nk_3 = "vision_model.head.attention.v_proj_bias"
                    nv_3 = nv[2*dim:]
                    new_state_dict[nk_1] = nv_1
                    new_state_dict[nk_2] = nv_2
                    new_state_dict[nk_3] = nv_3
                else:
                    new_state_dict[nk] = nv
                    
        state_dict = new_state_dict
        for name, data in state_dict.items():
            if should_skip_tensor(name, has_text_encoder, has_vision_encoder, has_xgenmm_projector):
                # we don't need this
                print(f"skipping parameter: {name}")
                continue

            name = get_tensor_name(name)
            data = data.squeeze().numpy()
            
            n_dims = len(data.shape)

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
        
        print("🟢 Vit tensors added !")    
        
    if has_xgenmm_projector:
        with print_time("Loading projector and converting to gguf"):
            projector_ckpt = torch.load(args.xgenmm_projector)
            projector = replace_tensor_name_xgenmm_projector(projector_ckpt)
            if args.use_f32:
                ftype = 0
            else:
                ftype = 1
            ftype_cur = ftype
            for name, tensor in projector.items():
                tensor = tensor.squeeze().numpy()
                if ftype_cur == 1:
                    if 'ln.bias' in name or 'ln.weight' in name:
                        tensor = tensor.astype(np.float32)
                        ftype_cur = 0
                        print(f'❗ {name} is set to np.float32')
                    else:
                        tensor = tensor.astype(np.float16)
                        ftype_cur = 1
                        print(f'❗ {name} is set to np.float16')
                else:
                    if tensor.dtype != np.float32:
                        tensor = tensor.astype(np.float32)
                        print(f'❗ {name} is set to np.float32')
                        ftype_cur = 0

                print(f"{name} - {ftype_str[ftype_cur]} - shape = {tensor.shape}")
                fout.add_tensor(name, tensor)
            print("🟢 Projector tensors added\n")
    
    with print_time("write to gguf file"):
        fout.write_header_to_file()
        fout.write_kv_data_to_file()
        fout.write_tensors_to_file()
        fout.close()
        print("🟢 Done. Output file: " + fname_out)