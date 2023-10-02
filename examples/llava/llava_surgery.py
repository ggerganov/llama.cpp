import argparse
from llava.model import LlavaLlamaForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel
import torch

dtype = torch.bfloat16

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to LLaVA RLHF model")
ap.add_argument("-o", "--output", help="Output directory to save the merged file")
args = ap.parse_args()

model_path = f"{args.model}/sft_model"
lora_path = f"{args.model}/rlhf_lora_adapter_model"
save_path = args.output

model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    device_map={"": "cuda:0"},
    torch_dtype=dtype,
)
model = PeftModel.from_pretrained(
    model,
    lora_path,
)


model = model.merge_and_unload()

model.save_pretrained(save_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_path)

del model
del tokenizer


# Load the checkpoint
checkpoint = torch.load(f"{save_path}/pytorch_model-00002-of-00002.bin")

# Extract the tensors we want
mm_projector_weight = checkpoint['model.mm_projector.weight']
mm_projector_bias = checkpoint['model.mm_projector.bias']

# Remove the tensors from the checkpoint
del checkpoint['model.mm_projector.weight']
del checkpoint['model.mm_projector.bias']

# Create a dictionary with the original names as keys
mm_projector = {
    'model.mm_projector.weight': mm_projector_weight,
    'model.mm_projector.bias': mm_projector_bias
}

# Save the combined dictionary using torch.save
torch.save(mm_projector, "projector.pt")

# Save the rest of the model with the same original name
torch.save(checkpoint, "./llava-7b-rlhf-merged/pytorch_model-00002-of-00002.bin")

Print("Operation complete!")
