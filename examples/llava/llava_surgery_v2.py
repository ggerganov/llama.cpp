import argparse
import glob
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Any, ContextManager, cast

# Function to determine if file is a SafeTensor file
def is_safetensor_file(file_path):
    return file_path.endswith('.safetensors')


# Unified loading function
def load_model(file_path):
    if is_safetensor_file(file_path):
        tensors = {}
        with cast(ContextManager[Any], safe_open(file_path, framework="pt", device="cpu")) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).clone()
                # output shape
                print(f"{key} : {tensors[key].shape}")
        return tensors, 'safetensor'
    else:
        return torch.load(file_path, map_location=torch.device('cpu')), 'pytorch'


# Unified saving function
def save_model(model, file_path, file_type):
    if file_type == 'safetensor':
        # safe_save(model, file_path)
        save_file(model, file_path)
    else:
        torch.save(model, file_path)


# Adapted function to clean vision tower from checkpoint
def clean_vision_tower_from_checkpoint(checkpoint_path):
    checkpoint, file_type = load_model(checkpoint_path)
    # file_type = 'pytorch'
    model_path = os.path.dirname(checkpoint_path)
    print(f"Searching for vision tower tensors in {checkpoint_path}")
    clip_tensors = [k for k, v in checkpoint.items() if (k.startswith("model.vision_tower") or k.startswith("vit.") or k.startswith("vision_tower"))]

    if len(clip_tensors) > 0:
        print(f"Found {len(clip_tensors)} tensors to extract from {checkpoint_path}")
        # Adapted for file type
        clip_path = os.path.join(model_path, "llava.clip")

        if os.path.exists(clip_path):
            print(f"Loading existing llava.clip from {clip_path}")
            existing_clip, _ = load_model(clip_path)
        else:
            print(f"Creating new llava.clip at {clip_path}")
            existing_clip = {}
        # Update existing_clip with new tensors, avoid duplicates
        for name in clip_tensors:
            simple_name = name[name.index('vision_model.'):] if 'vision_model.' in name else name
            print(f"Adding {simple_name} to llava.clip")
            if simple_name not in existing_clip:
                existing_clip[simple_name] = checkpoint[name]

        # Save the updated clip tensors back to llava.clip
        save_model(existing_clip, clip_path, 'pytorch')

        # Remove the tensors from the original checkpoint
        for name in clip_tensors:
            del checkpoint[name]

        checkpoint_path = checkpoint_path
        return True
    return False

def find_relevant_checkpoints(checkpoint_paths, newline_criteria, projector):
    newline_checkpoint_path = None
    projector_checkpoint_path = None

    for path in checkpoint_paths:
        checkpoint, _ = load_model(path)
        if newline_criteria(checkpoint) and newline_checkpoint_path is None:
            newline_checkpoint_path = path
        if projector(checkpoint):
            projector_checkpoint_path = path

    return newline_checkpoint_path, projector_checkpoint_path

def newline_criteria(checkpoint):
    return any(k.startswith("model.image_newline") or k.startswith("image_newline") for k in checkpoint.keys())

def proj_criteria(checkpoint):
    return any(k.startswith("model.mm_projector") or k.startswith("vision_proj.") or k.startswith("multi_modal_projector") for k in checkpoint.keys())


# Command-line interface setup
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to LLaVA v1.5+ model")
ap.add_argument("-C", "--clean-vision-tower", action="store_true", help="Remove any vision tower from the model files")
args = ap.parse_args()

if args.clean_vision_tower:
    # Generalized to handle both PyTorch and SafeTensors models
    model_files = sorted(glob.glob(f"{args.model}/*"), key=os.path.getmtime, reverse=True)
    # checkpoint_paths = [path for path in model_files if (path.endswith('.bin') and path.startswith('pytorch')) or (path.endswith('.safetensors') and path.startswith('model'))]
    checkpoint_paths = [path for path in model_files if (path.endswith('.bin') and 'pytorch' in path.split('/')[-1].split('\\')[-1]) or (path.endswith('.safetensors') and 'model' in path.split('/')[-1].split('\\')[-1])]
    for projector_checkpoint_path in checkpoint_paths:
        print(f"Cleaning {projector_checkpoint_path}")
        if not clean_vision_tower_from_checkpoint(projector_checkpoint_path):
            print(f"No vision tower found in {projector_checkpoint_path}")
            # we break once none is found, so far all models append them at the end
            # break
    print("Done! All vision tower tensors are removed from the model files and stored in llava.clip file.")

# Now we look for the projector in the last checkpoint
model_files = sorted(glob.glob(f"{args.model}/*"), key=os.path.getmtime, reverse=True)
checkpoint_paths = [path for path in model_files if (path.endswith('.bin') and 'pytorch' in path.split('/')[-1].split('\\')[-1]) or (path.endswith('.safetensors') and 'model' in path.split('/')[-1].split('\\')[-1])]
# last_checkpoint_path = checkpoint_paths[0]
# first_checkpoint_path = checkpoint_paths[-1]
newline_checkpoint_path, projector_checkpoint_path = find_relevant_checkpoints(checkpoint_paths, newline_criteria, proj_criteria)

print(f"Taking projector from {projector_checkpoint_path}")
first_mm_tensors = []
first_checkpoint = None
if newline_checkpoint_path is not None:
    print(f"Taking newline from {newline_checkpoint_path}")
    first_checkpoint, file_type = load_model(newline_checkpoint_path)
    first_mm_tensors = [k for k, v in first_checkpoint.items() if k.startswith("model.image_newline") or k.startswith("image_newline")]

# Load the checkpoint
mm_tensors = []
last_checkpoint = None
if projector_checkpoint_path is not None:
    last_checkpoint, file_type = load_model(projector_checkpoint_path)
    mm_tensors = [k for k, v in last_checkpoint.items() if k.startswith("model.mm_projector") or k.startswith("vision_proj.") or k.startswith("multi_modal_projector")]

if len(mm_tensors) == 0:
    if last_checkpoint is not None:
        for k, v in last_checkpoint.items():
            print(k)
    print(f"Found {len(mm_tensors)} tensors to extract out of {len(last_checkpoint) if last_checkpoint is not None else 0} tensors.")
    print("No tensors found. Is this a LLaVA model?")
    exit()

print(f"Found {len(mm_tensors)} tensors to extract.")
print(f"Found additional {len(first_mm_tensors)} tensors to extract.")
# projector = {name: checkpoint.[name].float() for name in mm_tensors}
projector = {}
for name in mm_tensors:
    assert last_checkpoint is not None
    # HACK - this should probably be in the second script...
    new_name = name
    if new_name.startswith("multi_modal_projector.linear_1"):
        new_name = new_name.replace("multi_modal_projector.linear_1", "mm.0")
    elif new_name.startswith("multi_modal_projector.linear_2"):
        new_name = new_name.replace("multi_modal_projector.linear_2", "mm.2")
    projector[new_name] = last_checkpoint[name].float()
for name in first_mm_tensors:
    assert first_checkpoint is not None
    # HACK - this should probably be in the second script too...
    new_name = name
    if new_name == "image_newline":
        new_name = "model.image_newline"
    projector[new_name] = first_checkpoint[name].float()

if len(projector) > 0:
    save_model(projector, f"{args.model}/llava.projector", 'pytorch')

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.")
