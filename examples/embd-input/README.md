### Examples for input embedding directly

## Requirement
build  `libembdinput.so`
run the following comman in main dir (../../).
```
make
```

## [LLaVA](https://github.com/haotian-liu/LLaVA/) example  (llava.py)

1. Obtian LLaVA model (following https://github.com/haotian-liu/LLaVA/ , use https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/).
2. Convert it to ggml format.
3. `llava_projection.pth` is [pytorch_model-00003-of-00003.bin](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/blob/main/pytorch_model-00003-of-00003.bin).

```
import torch

bin_path = "../LLaVA-13b-delta-v1-1/pytorch_model-00003-of-00003.bin"
pth_path = "./examples/embd-input/llava_projection.pth"

dic = torch.load(bin_path)
used_key = ["model.mm_projector.weight","model.mm_projector.bias"]
torch.save({k: dic[k] for k in used_key}, pth_path)
```
4. Check the path of LLaVA model and `llava_projection.pth` in `llava.py`.


## [PandaGPT](https://github.com/yxuansu/PandaGPT) example (panda_gpt.py)

1. Obtian PandaGPT lora model from https://github.com/yxuansu/PandaGPT. Rename the file to `adapter_model.bin`. Use [convert-lora-to-ggml.py](../../convert-lora-to-ggml.py) to convert it to ggml format.
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
2. Papare the `vicuna` v0 model.
3. Obtain the [ImageBind](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) model.
4. Clone the PandaGPT source.
```
git clone https://github.com/yxuansu/PandaGPT
```
5. Install the requirement of PandaGPT.
6. Check the path of PandaGPT source, ImageBind model, lora model and vicuna model in panda_gpt.py.

## [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4/) example (minigpt4.py)

1. Obtain MiniGPT-4 model from https://github.com/Vision-CAIR/MiniGPT-4/ and put it in `embd-input`.
2. Clone the MiniGPT-4 source.
```
git clone https://github.com/Vision-CAIR/MiniGPT-4/
```
3. Install the requirement of PandaGPT.
4. Papare the `vicuna` v0 model.
5. Check the path of MiniGPT-4 source, MiniGPT-4 model and vicuna model in `minigpt4.py`.
