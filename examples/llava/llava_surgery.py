import argparse
import glob
import os
import torch


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to LLaVA v1.5 model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
path = sorted(glob.glob(f"{args.model}/pytorch_model*.bin"))[-1]
checkpoint = torch.load(path)

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith("model.mm_projector")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name] for name in mm_tensors}
torch.save(projector, f"{args.model}/llava.projector")

# remove these tensors from the checkpoint and save it again
for name in mm_tensors:
    del checkpoint[name]

torch.save(checkpoint, path)

print("Done!")
print(f"Now you can convert {args.model} to a a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.")
