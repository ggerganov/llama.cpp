# InternVL

Currently this implementation supports [Mini-InternVL-Chat-2B-V1-5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5).

## Usage
Build with cmake or run `make llama-internvl-cli` to build it.

After building, run: `./llama-internvl-cli` to see the usage. For example:

```sh
./llama-internvl-cli -m InternVL-gguf/internlm2-1.8B-chat-q4_k.gguf --mmproj InternVL-gguf/InternViT-300M-448px-f16.gguf --image path/to/an/image.jpg -p "<image>\nPlease describe the image shortly."
```

## Model conversation

1. Clone `Mini-InternVL-Chat-2B-V1-5` locally:

```sh
git clone https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5
```

2. Copy `config.json` from [internlm2-chat-1_8b](https://huggingface.co/internlm/internlm2-chat-1_8b):
   

3. Use `split_language_tensors.py` to get LLaMA constituents:

```sh
mkdir adjusted-internlm-chat
python split_language_tensors.py -m path/to/Mini-InternVL-Chat-2B-V1-5/model.safetensors -o path/to/adjusted-internlm-chat/model.safetensors
```

4. Prepare the essentials for converting language model:
```sh
cp path/to/Mini-InternVL-Chat-2B-V1-5/*.json path/to/adjusted-internlm-chat/
cp path/to/Mini-InternVL-Chat-2B-V1-5/tokenizer.model path/to/adjusted-internlm-chat/
cp path/to/internlm2-chat-1_8b/config.json path/to/adjusted-internlm-chat/
```

5. Use `convert_hf_to_gguf.py` to convert the language model to GGUF:

```sh
python convert_hf_to_gguf.py path/to/adjusted-internlm-chat/
```

6. Use `vision_model_to_gguf.py` to convert the image encoder to GGUF:

```sh
python vision_model_to_gguf.py path/to/Mini-InternVL-Chat-2B-V1-5/model.safetensors
```

7. Collect and rename the models:
```sh
mkdir InternVL-gguf
mv Mini-InternVL-Chat-2B-V1-5/model.safetensors-f16.gguf InternVL-gguf/InternViT-300M-448px-f16.gguf
mv adjusted-internlm-chat/adjusted-internlm-1.9B-chat-F16.gguf InternVL-gguf/internlm2-1.8B-chat-F16.gguf
```

8. Use `llama-quantize` to convert the language model from `fp16` to `q4_k`
```sh
./llama-quantize path/to/InternVL-gguf/internlm2-1.8B-chat-F16.gguf path/to/InternVL-gguf/internlm2-1.8B-chat-q4_k.gguf q4_k_s
```

## Some result on Android with `Snapdragon 888+` chip
### case 1
**input**
```sh
/data/local/tmp/llama-internvl-cli \
    -m /data/local/tmp/internlm2-1.8B-chat-q4_k.gguf \
    --mmproj /data/local/tmp/InternViT-300M-448px-f16.gguf \
    -t 4 \
    --image /data/local/tmp/image1.jpg \
    -p "<image>\nPlease describe the image shortly." \
    -b 4096 -c 4096
```

**output**
```sh
encode_image_with_clip: image embedding created: 1792 tokens

encode_image_with_clip: image encoded in 164683.39 ms by CLIP (   91.90 ms per image patch)
The image shows a young red panda peeking over the top of a wooden platform or platform-like structure, with its head sticking out over the edge. The panda has a striking red coat with white patches around its eyes and ears. Its fur looks fluffy and it has a black nose and mouth. The background is green and blurry, suggesting it might be an outdoor setting, possibly a zoo or a sanctuary. The wood on the platform looks worn and worn, indicating it might be well used.
llama_print_timings:        load time =  316889.60 ms
llama_print_timings:      sample time =       7.27 ms /   103 runs   (    0.07 ms per token, 14173.66 tokens per second)
llama_print_timings: prompt eval time =  151858.76 ms /  1831 tokens (   82.94 ms per token,    12.06 tokens per second)
llama_print_timings:        eval time =   19437.72 ms /   102 runs   (  190.57 ms per token,     5.25 tokens per second)
llama_print_timings:       total time =  336547.70 ms /  1933 tokens
```

### case2
**input**
```sh
/data/local/tmp/llama-internvl-cli \
    -m /data/local/tmp/internlm2-1.8B-chat-q4_k.gguf \
    --mmproj /data/local/tmp/InternViT-300M-448px-f16.gguf \
    -t 4 \
    --image /data/local/tmp/demo.jpg \
    -p "<image>\nWho is the author of this book? \nAnswer the question using a single word or phrase."
```

**output**
```sh
encode_image_with_clip: image embedding created: 768 tokens

encode_image_with_clip: image encoded in 87791.64 ms by CLIP (  114.31 ms per image patch)
Susan Wise Bauer
llama_print_timings:        load time =  144433.03 ms
llama_print_timings:      sample time =       0.51 ms /     6 runs   (    0.08 ms per token, 11834.32 tokens per second)
llama_print_timings: prompt eval time =   55674.58 ms /   820 tokens (   67.90 ms per token,    14.73 tokens per second)
llama_print_timings:        eval time =     581.98 ms /     5 runs   (  116.40 ms per token,     8.59 tokens per second)
llama_print_timings:       total time =  145118.73 ms /   825 tokens
```

## Running on Nvidia 4090
### case1

**input**
```sh
bin/llama-internvl-cli \
    -m path/to/internlm2-1.8B-chat-q4_k.gguf \
    --mmproj path/to/InternViT-300M-448px-f16.gguf \
    -t 4 --image path/to/image1.jpg \
    -p "<image>\nPlease describe the image shortly." \
    --gpu-layers 1000 -b 4096 -c 4096
```

**output**
```sh
encode_image_with_clip: image embedding created: 1792 tokens

encode_image_with_clip: image encoded in   278.86 ms by CLIP (    0.16 ms per image patch)

The image depicts a red panda, a small, wild bear found primarily in the mountains of central and south-eastern China, and the surrounding areas of India and Nepal. This species is distinguished by its distinctive red fur with a white face and legs, which is a unique feature that helps them blend in with their natural habitat of mountainous regions. The red panda is known for its thick fur, which is typically a blend of red, black, and white fur, with a thick tail and large ears, which aid in thermoregulation.

In the image, the red panda is leaning over a wooden platform, which appears to be part of a man-made enclosure, likely within a zoo or wildlife park. The platform is made of wood and seems to be constructed to provide the animals with a place to rest or interact with visitors. The background features trees with green foliage, indicating that the setting is an outdoor environment with ample vegetation, which is typical for red panda habitats.

The red panda’s front paws are resting on the platform, while its head is slightly tilted, giving an impression that it is engaged with something in front of it, possibly a camera, a person, or an object placed on the platform. Its eyes are large and dark, which are characteristic of the species, and it has a slightly wrinkled face, typical of many bear species, which helps them stay warm in cold temperatures. The expression on the panda’s face appears curious or attentive, as it looks directly at the camera or observer.

In summary, the image showcases a red panda in an outdoor setting with a wooden platform, surrounded by green trees. The animal appears to be relaxed and engaged, likely interacting with the observer or something placed on the platform. The setting suggests that the panda is in a zoo or a wildlife sanctuary, where it is cared for and protected from the wild.

llama_print_timings:        load time =     723.77 ms
llama_print_timings:      sample time =      14.28 ms /   392 runs   (    0.04 ms per token, 27443.29 tokens per second)
llama_print_timings: prompt eval time =     107.43 ms /  1831 tokens (    0.06 ms per token, 17043.81 tokens per second)
llama_print_timings:        eval time =    1184.80 ms /   391 runs   (    3.03 ms per token,   330.01 tokens per second)
llama_print_timings:       total time =    1942.12 ms /  2222 tokens
```

### case2
**input**
```sh
/data/local/tmp/llama-internvl-cli \
    -m /data/local/tmp/internlm2-1.8B-chat-q4_k.gguf \
    --mmproj /data/local/tmp/InternViT-300M-448px-f16.gguf \
    -t 4 \
    --image /data/local/tmp/demo.jpg \
    -p "<image>\nWho is the author of this book? \nAnswer the question using a single word or phrase." \
    --gpu-layers 1000
```

**output**
```sh
encode_image_with_clip: image embedding created: 768 tokens

encode_image_with_clip: image encoded in   138.85 ms by CLIP (    0.18 ms per image patch)

Susan Wise Bauer

llama_print_timings:        load time =     430.77 ms
llama_print_timings:      sample time =       0.21 ms /     6 runs   (    0.03 ms per token, 28571.43 tokens per second)
llama_print_timings: prompt eval time =      70.31 ms /   820 tokens (    0.09 ms per token, 11661.97 tokens per second)
llama_print_timings:        eval time =      15.84 ms /     5 runs   (    3.17 ms per token,   315.68 tokens per second)
llama_print_timings:       total time =     446.85 ms /   825 tokens
```