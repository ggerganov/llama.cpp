# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**.

**WORK IN PROGRESS!** **Status**: INT4 gives not so good quality, need to properly measure and compare perplexity.

## Plan

1. Create Python script with sampling and simple chat interface
2. Measure performance and quality of different model sizes and data types
3. Write a good `README.md` and publish links to this repo
4. Create pull request to main `ggml` repo with all improvements made here

## Structure

- `./rwkv.h`, `./rwkv.cpp`: source code of the shared library.
- `./rwkv`: directory containing Python scripts for conversion, inference and validation.

## How to use

### Windows

Requirements: [git](https://gitforwindows.org/), [CMake](https://cmake.org/download/), MSVC compiler, Python 3.x with PyTorch.

#### 1. Clone the repo and build it:

```commandline
git clone https://github.com/saharNooby/rwkv.cpp.git
cd rwkv.cpp
cmake -DBUILD_SHARED_LIBS=ON .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

#### 2. Download an RWKV model from [Huggingface](https://huggingface.co/BlinkDL) and convert it into `ggml` format:

```commandline
python rwkv\convert_pytorch_rwkv_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
```

#### 3. Use the model in Python:

```python
# These files are located in rwkv directory
import rwkv_cpp_model
import rwkv_cpp_shared_library

model = rwkv_cpp_model.RWKVModel(
    rwkv_cpp_shared_library.load_rwkv_shared_library(),
    r'C:\rwkv.cpp-169M.bin'
)

logits, state = None, None

for token in [1, 2, 3]:
    logits, state = model.eval(token, state)
    
    print(f'Output logits: {logits}')

# Don't forget to free the memory after you've done working with the model
model.free()

```
