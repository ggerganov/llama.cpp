# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml). The end goal is to allow 4-bit quanized inference on CPU.

**WORK IN PROGRESS!** **Status**: There is a Python wrapper, FP32 and FP16 inference work correctly. Currently, I'm working on INT4 quantization support.

## Plan

1. Make INT4 inference work
2. Create Python script with sampling and simple chat interface
3. Clean up the repo (remove llama related files and mentions)
4. Write a good `README.md` and publish links to this repo
5. Create pull request to main `ggml` repo with all improvements made here

## Structure

This repo is based on the [llama.cpp repo](https://github.com/ggerganov/llama.cpp). RWKV-related code is in these directories:

- `./rwkv`: directory containing Python scripts for conversion, inference and validation
- `./examples/main_rwkw`: directory containing script that loads and infers RWKV model

Please do not change files in other directories — this will make pulling recent changes easier.

## How to use

### Windows

Requirements: [git](https://gitforwindows.org/), [CMake](https://cmake.org/download/), MSVC compiler, Python 3.x with PyTorch.

#### 1. Clone the repo and build it:

```commandline
git clone https://github.com/saharNooby/rwkv.cpp.git
cd rwkv.cpp
cmake -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

#### 2. Download an RWKV model from [Huggingface](https://huggingface.co/BlinkDL) and convert it into `ggml` format:

```commandline
python rwkv\convert_pytorch_rwkv_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
```

#### 3. Use the model in Python:

```python
# This file is located at rwkv/rwkv_cpp.py
import rwkv_cpp

model = rwkv_cpp.RWKVModel(r'bin\Release\rwkv.dll', r'C:\rwkv.cpp-169M.bin')

logits, state = None, None

for token in [1, 2, 3]:
    logits, state = model.eval(token, state)
    
    print(f'Output logits: {logits}')

# Don't forget to free memory after you've done working with the model
model.free()

```
