# omni-vlm

Currently this implementation supports [omni-vlm](https://huggingface.co/NexaAIDev/nano-vlm-instruct) variants,

After API is confirmed, more models will be supported / uploaded.

## Usage
Build with cmake in the `llama-cpp-experiments` folder:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --verbose -j
```
After building, run: `./omni-vlm-cli` to see the usage. For example:

```bash
./omni-vlm-cli \
    -m Nano-Llm-494M-F16.gguf \
    --mmproj mmproj-omni-vlm-f16.gguf \
    --image example/omni-vlm/cat.png
```

See next section to convert gguf files from original safetensors.

[comment]: # (TODO:
**note**: A lower temperature like 0.1 is recommended for better quality. add `--temp 0.1` to the command to do so.
**note**: For GPU offloading ensure to use the `-ngl` flag just like usual
)

## Omni-vlm gguf conversion
1) First clone omni-vlm model:
```console
git clone https://huggingface.co/NexaAIDev/nano-vlm-instruct
```

2) Install the required Python packages:

```sh
pip install -r examples/omni-vlm/requirements.txt
```

3) Run `omni_vlm_surgery.py`:
```console
python omni_vlm_surgery.py \
  --clean-vision-tower \
  --model <PATH TO nano-vlm-instruct>
```
- you will find an `omni_vlm.projector` and an `omni_vlm.clip` file in `nano-vlm-instruct/` directory

4) Create a soft link `pytorch_model.bin` to `omni_vlm.clip`:
```bash
# in nano-vlm-instruct/ folder
ln -s omni_vlm.clip pytorch_model.bin
```
5) Go back to `llama.cpp` project folder and create the visual gguf model:

clone `nano-vlm-processor` model directory (You may need to obtain authorization to access NexaAIDev space).
```console
git clone https://huggingface.co/NexaAIDev/nano-vlm-processor
```

```console
python ./examples/omni-vlm/convert_image_encoder_to_gguf.py \
    -m <PATH TO nano-vlm-instruct> \
    --output-dir <PATH TO nano-vlm-instruct> \
    -p <PATH TO nano-vlm-processor>
```
- You will get a pure vision model part of CLIP named `<PATH TO nano-vlm-instruct>/mmproj-omni-vlm-f16.gguf`.

6) Then convert the LLM portion to gguf format:
* Run python snippet below to extract LLM portion from original omni-vlm model.
```python
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {}
with safe_open("<PATH TO nano-vlm-instruct>/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        if k.startswith('language_model'):
            k2 = k.replace('language_model.', '')
            tensors[k2] = f.get_tensor(k)

    save_file(tensors, "<PATH TO nano-vlm-processor>/model.safetensors")
```

```console
python convert_hf_to_gguf.py <PATH TO nano-vlm-processor>
```
Finally we will get LLM GGUF model: `<PATH TO nano-vlm-processor>/Nano-Llm-494M-F16.ggu`

7) And finally we can run the omni-vlm demo of C++ version:
```console
./build/bin/omni-vlm-cli \
    -m  <PATH TO nano-vlm-processor>/Nano-Llm-494M-F16.gguf \
    --mmproj <PATH TO nano-vlm-instruct>/mmproj-omni-vlm-f16.gguf \
    --image example/omni-vlm/cat.png
```
The results will print on the screen:
> The image depicts a grey and white cat with its head pressed against the camera, appearing as if it is staring directly into the lens. The cat is surrounded by black and white stripes, adding a unique touch to its appearance. The black background creates a strong contrast and highlights the cat's features, making it a captivating scene.

8) Python interface:

After successfully compiling omni_vlm_wrapper_shared dynamic library, run:
```console
python omni_vlm_demo.py \
  --model <PATH TO nano-vlm-processor>/Nano-Llm-494M-F16.gguf \
  --mmproj <PATH TO nano-vlm-instruct>/mmproj-omni-vlm-f16.gguf \
  --prompt="Describe this image for me" \
  --image-path cat.png
```
