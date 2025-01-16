# Instructions to Convert Multimodal Granite -> GGUF
Disclaimer that this branch is super WIP; eventually this should be combined with the main README in this directory, but separating it for now since we use a different method for converting the LLM.

First, set the env var `$GRANITE_MODEL` to your vLLM/transformers format multimodal granite model.

`export GRANITE_MODEL=...`


### 1. Running llava surgery v2.
First, we need to run the llava surgery script as shown below:

`python llava_surgery_v2.py -C -m $GRANITE_MODEL`

You should see two new files (`llava.clip` and `llava.projector`) written into your model's directory. You can load them directly with pytorch and validate that they are nonempty using the snippet below.

`ls $GRANITE_MODEL | grep -i llava`



We should see that the projector and visual encoder get split out into the llava files. Quick check to make sure they aren't empty:
```python
import os
import torch

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

encoder_tensors = torch.load(os.path.join(MODEL_PATH, "llava.clip"))
projector_tensors = torch.load(os.path.join(MODEL_PATH, "llava.projector"))

assert len(encoder_tensors) > 0
assert len(projector_tensors) > 0
```

If you actually inspect the `.keys()` of the loaded tensors, you should see a lot of `vision_model` tensors in the `encoder_tensors`, and 5 tensors (`'mm.0.bias'`, `'mm.0.weight'`, `'mm.2.bias'`, `'mm.2.weight'`, `'model.image_newline'`) in the multimodal `projector_tensors`.



### 2. Creating the Visual Component GGUF
To create the GGUF for the visual components, we need to write a config for the visual encoder. Here is an example Alex wrote for initial testing using the values in the preliminary model; if things are going wrong, there is a good chance it's a misalignment with the config here.

Note: we refer to this file as `$VISION_CONFIG` later on.
```json
{
    "_name_or_path": "siglip-model",
    "architectures": [
      "SiglipVisionModel"
    ],
    "image_grid_pinpoints": [
        [384,768],
        [384,1152],
        [384,1536],
        [384,1920],
        [384,2304],
        [384,2688],
        [384,3072],
        [384,3456],
        [384,3840],
        [768,384],
        [768,768],
        [768,1152],
        [768,1536],
        [768,1920],
        [1152,384],
        [1152,768],
        [1152,1152],
        [1536,384],
        [1536,768],
        [1920,384],
        [1920,768],
        [2304,384],
        [2688,384],
        [3072,384],
        [3456,384],
        [3840,384]
    ],
    "hidden_size": 1152,
    "image_size": 384,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "transformers_version": "4.45.0.dev0",
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
    "projection_dim": 0
}  
```

Create a new directory to hope the visual components, and copy the llava.clip/projector files, as well as the vision config into it.

```
ENCODER_PATH=...
VISION_CONFIG=...
mkdir $ENCODER_PATH

cp $GRANITE_MODEL/llava.clip $ENCODER_PATH/pytorch_model.bin
cp $GRANITE_MODEL/llava.projector $ENCODER_PATH/
cp $VISION_CONFIG $ENCODER_PATH/config.json
```

At which point you should have something like this:
```bash
(venv) alexanderjbrooks@wecm-9-67-137-179 llava % ls $ENCODER_PATH 
config.json             llava.projector         pytorch_model.bin
```

Now convert the components to GGUF.
```bash
python convert_image_encoder_to_gguf.py \
    -m $ENCODER_PATH \
    --llava-projector $ENCODER_PATH/llava.projector \
    --output-dir mgranite_siglip \
    --clip-model-is-vision \
    --clip-model-is-siglip
```

which will create the first GGUF file at `$ENCODER_PATH/mmproj-model-f16.gguf`; we will refer to the abs path of this file as the `$VISUAL_GGUF_PATH.`



### 3. Creating the LLM GGUF.
For now, the easiest way to get the GGUF for LLM is by loading the composite model in `transformers` and exporting the LLM so that it can be directly converted (Alex will add support to the converter for llava next if possible, but hacking to ignore unused tensors etc with the current instructions currently results in the tokenizer embedding weights not being found).

To do this, you can do something like the following; we assume you're setting the environment variable `LLM_EXPORT_PATH` to the place to put the exported `transformers` LLM.

```python
import os
import transformers

MODEL_PATH = os.getenv("GRANITE_MODEL")
if not MODEL_PATH:
    raise ValueError("env var GRANITE_MODEL is unset!")

LLM_EXPORT_PATH = os.getenv("LLM_EXPORT_PATH")
if not MODEL_PATH:
    raise ValueError("env var LLM_EXPORT_PATH is unset!")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
# NOTE: need to ignore mismatched sizes for now until Alex's mmgranite PR is merged in transformers
# This also causes actual problems for the load multimodal projector, but since we only
# export the LLM, we don't care for now...
model = transformers.AutoModelForImageTextToText.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)

tokenizer.save_pretrained(LLM_EXPORT_PATH)
model.language_model.save_pretrained(LLM_EXPORT_PATH)
```    

Now you can convert the exported LLM to GGUF with the normal converter.

```bash
LLM_GGUF_PATH=...

python convert_hf_to_gguf.py --outfile $LLM_GGUF_PATH $LLM_EXPORT_PATH
```



### 4. Running the model in llama cpp
Build llama cpp normally; you should have a target binary named `llama-llava-cli`, which you can pass two binaries to. Sample usage:
```
./build/bin/llama-llava-cli -m $LLM_GGUF_PATH \
    --mmproj $VISUAL_GGUF_PATH \
    --image cherry_blossom.jpg \
    -c 16384 \
    -p "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n\<image>\nCan you describe this image?\n<|assistant|>\n"
```

