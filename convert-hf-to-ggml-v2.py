import io
import os
import sys
import struct
import json
import torch
import numpy as np
import tempfile
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig

conv_map = {
    'word_embeddings': 'tok_embeddings',
    'word_embeddings_layernorm': 'norm',
    'input_layernorm': 'attention_norm',
    'self_attention.query_key_value': 'attention.query_key_value',
    'self_attention.dense': 'attention.wo',
    'post_attention_layernorm': 'ffn_norm',
    'mlp.dense_h_to_4h': 'feed_forward.w1',
    'mlp.dense_4h_to_h': 'feed_forward.w2',
    'ln_f': 'output_norm',
    'lm_head': 'output',
}

parser = argparse.ArgumentParser(description='Convert a model from HF format to GGML format.')
parser.add_argument('model_name', type=str, help='directory of the model to convert. Example: "bigscience/bloomz-560m"')
parser.add_argument('dir_output', type=str, help='directory where the output file will be written')
parser.add_argument('--use-f32', action='store_true', help='if present, use float32 instead of float16')
parser.add_argument('--debug', action='store_true', help='if present, dump the progress as it happens')
args = parser.parse_args()

model_name = args.model_name
dir_out = args.dir_output

os.makedirs(dir_out, exist_ok=True)

ftype_str = ["f32", "f16"]
ftype = 0 if args.use_f32 else 1
debug_flag = args.debug

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
hparams = config.to_dict()
print("Loading model: ", model_name)

# Save the model to disk
model_dir = f"{model_name}_tmp"
os.makedirs(model_dir, exist_ok=True)
config.save_pretrained(model_dir)

fname_out = dir_out + f"/ggml-model-{model_name.split('/')[-1]}-{ftype_str[ftype]}.bin"
fout = open(fname_out, "wb")

hparams["multiple_of"] = 1
fout.write(struct.pack("i", 0x67676d6c))
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["multiple_of"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", ftype))

dot_token = tokenizer.encode(".")[0]
for i in range(hparams["vocab_size"]):
    text = tokenizer.decode([i]).encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# Create temporary files for chunks
temp_files = {}

# Define the chunk size
chunk_size = 1000 * 1000 * 1024

# Load the PyTorch model weights from the saved files
# Find the files in the model directory
model_files = sorted([f for f in os.listdir(model_name) if f.startswith("pytorch_model") and f.endswith(".bin")])

added_head = False
state_dict = {}
for model_file in tqdm(model_files, desc="Processing model files in: " + model_name):
    file_path = os.path.join(model_name, model_file)
    model_part = torch.load(file_path, map_location=torch.device('cpu'))
    state_dict.update(model_part)

    # Add the missing lm_head.weight tensor
    lm_head_weight_key = 'lm_head.weight'
    word_embeddings_weight_key = 'word_embeddings.weight'
    if lm_head_weight_key not in state_dict and not added_head:
        # Use the word_embeddings.weight tensor for the lm_head.weight
        word_embeddings_weight = state_dict[word_embeddings_weight_key]

        # Add the tensor to the state_dict
        state_dict[lm_head_weight_key] = word_embeddings_weight

        added_head = True


    for name in tqdm(state_dict.keys(), desc="Processing nodes"):
        src = name
        nn = name.split(".")
            
        # Handle layer indices
        if nn[0].isdigit():
            layer_idx = nn[0]
            nn = nn[1:]
        else:
            layer_idx = None

        if debug_flag:
            if nn[0].isdigit():
                print("For Layer: " + layer_idx)

        if nn[0] == "h":
            nn[0] = "layers"
            mapped = conv_map[".".join(nn[2:-1])]
            if layer_idx is not None:
                name = f"{layer_idx}.{nn[0]}.{layer_idx}.{mapped}.{nn[-1]}"
            else:
                name = ".".join(nn[:2] + [mapped] + nn[-1:])
        else:
            mapped = conv_map[".".join(nn[:-1])]
            if layer_idx is not None:
                name = f"{layer_idx}.{mapped}.{nn[-1]}"
            else:
                name = ".".join([mapped] + nn[-1:])

        if "query_key_value" in src:
            q, k, v = state_dict[src].reshape(config.n_head, 3, -1).unbind(1)
            state_dict[src] = torch.cat([q, k, v], dim=0).reshape_as(state_dict[src])

        if debug_flag:
            print(src, ' -> ', name)
        tensor = state_dict[src].cpu()
        
        # If the tensor dtype is bfloat16, convert it to float32
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        
        data = tensor.squeeze().numpy()
        data = data.astype(np.float32)

        n_dims = len(data.shape)

        if debug_flag:
            print(name, n_dims, data.shape)

        # Check if the current data type is float16
        if data.dtype == np.float16:
            ftype_cur = 1
        else:
            ftype_cur = 0

        # If the specified ftype is float16 and the current ftype is not, convert data to float16
        if ftype == 1 and ftype_cur == 0 and n_dims > 1:
            if debug_flag:
                print("Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1

        str = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str)

        # Write data to file in chunks
        data_buffer = data.tobytes()
        data_len = len(data_buffer)
        for offset in range(0, data_len, chunk_size):
            chunk = data_buffer[offset: offset + chunk_size]
            fout.write(chunk)

    # Free some memory as we don't need the previous layer's state
    state_dict = {}

fout.close()
print("Done. Output file: " + fname_out)
print("")