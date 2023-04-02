# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides the usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**.

RWKV is a novel large language model architecture, [with the largest model in the family having 14B parameters](https://huggingface.co/BlinkDL/rwkv-4-pile-14b). In contrast to Transformer with `O(n^2)` attention, RWKV requires only state from previous step to calculate logits. This makes RWKV very CPU-friendly on large context lenghts.

This project provides [a C library rwkv.h](rwkv.h) and [a convinient Python wrapper](rwkv%2Frwkv_cpp_model.py) for it.

**TODO**:

1. Measure performance and perplexity of different model sizes and data types
2. Write a good `README.md` (motivation, benchmarks, perplexity) and publish links to this repo
3. Create pull request to main `ggml` repo with all improvements made here

## How to use

### 1. Clone the repo and build the library

### Windows

**Requirements**: [git](https://gitforwindows.org/), [CMake](https://cmake.org/download/), MSVC compiler.

```commandline
git clone https://github.com/saharNooby/rwkv.cpp.git
cd rwkv.cpp
cmake -DBUILD_SHARED_LIBS=ON .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

### 2. Download an RWKV model from [Hugging Face](https://huggingface.co/BlinkDL) and convert it into `ggml` format

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/).

```commandline
python rwkv\convert_pytorch_rwkv_to_ggml.py C:\RWKV-4-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
```

#### 2.1. Optionally, quantize the model

To convert the model into INT4 quantized format, run:

```commandline
python rwkv\quantize.py C:\rwkv.cpp-169M.bin C:\rwkv.cpp-169M-Q4_1.bin 3
```

Pass `2` for `Q4_0` format (smaller size, lower quality), `3` for `Q4_1` format (larger size, higher quality).

### 3. Run the model

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/) and [tokenizers](https://pypi.org/project/tokenizers/).

To generate some text, run:

```commandline
python rwkv\generate_completions.py C:\rwkv.cpp-169M.bin
```

To chat with a bot, run:

```commandline
python rwkv\chat_with_bot.py C:\rwkv.cpp-169M.bin
```

Edit [generate_completions.py](rwkv%2Fgenerate_completions.py) or [chat_with_bot.py](rwkv%2Fchat_with_bot.py) to change prompts and sampling settings.

---

Example of using `rwkv.cpp` in your custom Python script:

```python
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
