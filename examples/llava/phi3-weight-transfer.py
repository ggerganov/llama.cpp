import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def main(args):

    # https://stackoverflow.com/questions/67689219/copy-one-layers-weights-from-one-huggingface-bert-model-to-another

    phi3_vision = AutoModelForCausalLM.from_pretrained(args.phi3v_base_path,\
                                                    device_map="auto",\
                                                    trust_remote_code=True,\
                                                    torch_dtype=torch.float16,\
                                                    _attn_implementation='eager')

    print("PHI3 VISION LOADED IN MEMORY")

    phi3_base = AutoModelForCausalLM.from_pretrained(args.phi3_instruct_base_path,\
                                                    device_map="auto",\
                                                    trust_remote_code=True,\
                                                    torch_dtype=torch.float16,\
                                                    _attn_implementation='eager')

    print("PHI3 BASE LOADED IN MEMORY")

    phi3_vision_layers = dict(phi3_vision.named_parameters())
    phi3_base_layers = dict(phi3_base.named_parameters())

    parts = list(set(phi3_vision_layers.keys()) & set(phi3_base_layers.keys()))

    print("----------------------------------------------------")
    print("before transfer")
    print(dict(phi3_vision.named_parameters())["model.layers.19.mlp.gate_up_proj.weight"] == \
    dict(phi3_base.named_parameters())["model.layers.19.mlp.gate_up_proj.weight"])
    print("----------------------------------------------------")

    for part in parts:
        phi3_base_layers[part].data.copy_(phi3_vision_layers[part].data)  
        # target                           # source

    print("----------------------------------------------------")
    print("after transfer")
    print(dict(phi3_vision.named_parameters())["model.layers.19.mlp.gate_up_proj.weight"] == \
    dict(phi3_base.named_parameters())["model.layers.19.mlp.gate_up_proj.weight"])
    print("----------------------------------------------------")

    # save updated model weights 
    outfile = "phi3-instruct-vision-weight-transfer.safetensors"
    outpath = os.path.join(args.phi3_instruct_base_path, outfile)
    save_file(phi3_base_layers, outpath)
    print(f"updates .safetensors saved to {outpath}")

    # update safetensors index config
    weight_index_path = os.path.join(args.phi3_instruct_base_path, "model.safetensors.index.json")

    with open(weight_index_path, "r") as f:
        index_data = json.load(f)
    
    for k,v in index_data["weight_map"].items():
        if v != "phi3-instruct-vision-weight-transfer.safetensors":
            index_data["weight_map"][k] = outfile

    with open(weight_index_path, "w") as f:
        json.dump(index_data, f)

    print(f"hf saftensor mapping updated!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="script to copy weights from PHI3V language model to PHI3-instruct")

    parser.add_argument("--phi3-instruct-base-path", type=str, default="microsoft/Phi-3-mini-128k-instruct", help="model path or model card for  PHI3-instruct")
    parser.add_argument("--phi3v-base-path", type=str, default="microsoft/Phi-3-vision-128k-instruct", help="model path or model card for PHI3V")

    main(parser.parse_args())
