### Examples for input embedding directly

## Requirement 
build  `libembd_input.so`
run the following comman in main dir (../../).
```
make
```

## LLAVA example  (llava.py)

1. obtian llava model (following https://github.com/haotian-liu/LLaVA/ , use https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/)
2. convert it to ggml format
3. llava_projection.pth is [pytorch_model-00003-of-00003.bin](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/blob/main/pytorch_model-00003-of-00003.bin)

```
import torch

bin_path = "../LLaVA-13b-delta-v1-1/pytorch_model-00003-of-00003.bin"
pth_path = "./examples/embd_input/llava_projection.pth"

dic = torch.load(bin_path)
used_key = ["model.mm_projector.weight","model.mm_projector.bias"]
torch.save({k: dic[k] for k in used_key}, pth_path)
```

## PandaGPT example (panda_gpt.py)

1. Obtian PandaGPT lora model. Rename the file to `adapter_model.bin`. Use [convert-lora-to-ggml.py](../../convert-lora-to-ggml.py) to convert it to ggml format.
The `adapter_config.json` is
```
{
  "peft_type": "LORA",
  "fan_in_fan_out": false,
  "bias": null,
  "modules_to_save": null,
  "r": 32,
  "lora_alpha": 32,
  "lora_dropout": 0.1,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```
2. papare the `vicuna` v0 model.
3. obtain the [ImageBind](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) model.
4. Clone the PandaGPT source.
5. check the path of PandaGPT source, ImageBind model, lora model and vicuna model in panda_gpt.py.

