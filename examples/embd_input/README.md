### Examples for input embedding directly

## LLAVA example  (llava.py)

1. obtian llava model (following https://github.com/haotian-liu/LLaVA/ , use https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/)
2. build  `libembd_input.so`
```
make
```
3. convert it to ggml format
4. llava_projection.pth is [pytorch_model-00003-of-00003.bin](https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/blob/main/pytorch_model-00003-of-00003.bin)

```
import torch

bin_path = "../LLaVA-13b-delta-v1-1/pytorch_model-00003-of-00003.bin"
pth_path = "./examples/embd_input/llava_projection.pth"

dic = torch.load(bin_path)
used_key = ["model.mm_projector.weight","model.mm_projector.bias"]
torch.save({k: dic[k] for k in used_key}, pth_path)
```

