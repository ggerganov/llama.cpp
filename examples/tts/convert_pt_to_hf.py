# convert the https://huggingface.co/novateur/WavTokenizer-large-speech-75token to HF format
# the goal is to be able to reuse the convert_hf_to_gguf.py after that to create a GGUF file with the OuteTTSS vocoder
#
# TODO: this script is LLM-generated and probably very inefficient and should be rewritten

import torch
import json
import os
import sys
import re

from safetensors.torch import save_file

# change path to script dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# default
model_path = './model.pt';

# read from CLI
if len(sys.argv) > 1:
    model_path = sys.argv[1]

# get the directory of the input model
path_dst = os.path.dirname(model_path)

print(f"Loading model from {model_path}")

model = torch.load(model_path, map_location='cpu')

#print(model)

# print all keys
for key in model.keys():
    print(key)
    if key == 'hyper_parameters':
        #print(model[key])
        # dump as json pretty
        print(json.dumps(model[key], indent=4))
    #if key != 'state_dict' and key != 'optimizer_states':
    #    print(model[key])

# Check if the loaded model is a state_dict or a model instance
if isinstance(model, torch.nn.Module):
    state_dict = model.state_dict()
else:
    state_dict = model

# Print the structure of the state_dict to understand its format
print("State dictionary keys:")
for key in state_dict.keys():
    print(key)

# Ensure the state_dict is flat and contains only torch.Tensor objects
def flatten_state_dict(state_dict, parent_key='', sep='.'):
    items = []
    items_new = []

    for k, v in state_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, torch.Tensor):
            items.append((new_key, v))
        elif isinstance(v, dict):
            items.extend(flatten_state_dict(v, new_key, sep=sep).items())
            return dict(items)

    size_total_mb = 0

    for key, value in list(items):
        # keep only what we need for inference
        if not key.startswith('state_dict.feature_extractor.encodec.quantizer.') and \
           not key.startswith('state_dict.backbone.') and \
           not key.startswith('state_dict.head.'):
               print('Skipping key: ', key)
               continue

        new_key = key

        new_key = new_key.replace('state_dict.', '')

        # check if matches "backbone.pos_net.%d.bias" or "backbone.pos_net.%d.weight"
        if new_key.startswith("backbone.pos_net."):
            match = re.match(r"backbone\.pos_net\.(\d+)\.(bias|weight)", new_key)
            if match:
               new_key = f"backbone.pos_net.{match.group(1)}.norm.{match.group(2)}"

        size_mb = value.element_size() * value.nelement() / (1024 * 1024)
        print(f"{size_mb:8.2f} MB - {new_key}: {value.shape}")

        size_total_mb += size_mb

        #print(key, '->', new_key, ': ', value)
        #print(key, '->', new_key)

        items_new.append((new_key, value))

    print(f"Total size: {size_total_mb:8.2f} MB")

    return dict(items_new)

flattened_state_dict = flatten_state_dict(state_dict)


# Convert the model to the safetensors format
output_path = path_dst + '/model.safetensors'
save_file(flattened_state_dict, output_path)

print(f"Model has been successfully converted and saved to {output_path}")

# Calculate the total size of the .safetensors file
total_size = os.path.getsize(output_path)

# Create the weight map
weight_map = {
    "model.safetensors": ["*"]  # Assuming all weights are in one file
}

# Create metadata for the index.json file
metadata = {
    "total_size": total_size,
    "weight_map": weight_map
}

# Save the metadata to index.json
index_path = path_dst + '/index.json'
with open(index_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata has been saved to {index_path}")

config = {
    "architectures": [
        "OuteTTSVocoder"
    ],
    "num_hidden_layers": 12
}

with open(path_dst + '/config.json', 'w') as f:
    json.dump(config, f, indent=4)

print(f"Config has been saved to {path_dst + 'config.json'}")
