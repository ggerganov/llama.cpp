# rwkv.cpp

This is a port of [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) to [ggerganov/ggml](https://github.com/ggerganov/ggml).

Besides the usual **FP32**, it supports **FP16** and **quantized INT4** inference on CPU. This project is **CPU only**.

RWKV is a novel large language model architecture, [with the largest model in the family having 14B parameters](https://huggingface.co/BlinkDL/rwkv-4-pile-14b). In contrast to Transformer with `O(n^2)` attention, RWKV requires only state from previous step to calculate logits. This makes RWKV very CPU-friendly on large context lenghts.

This project provides [a C library rwkv.h](rwkv.h) and [a convinient Python wrapper](rwkv%2Frwkv_cpp_model.py) for it.

**TODO (contributions welcome!)**:

1. Measure latency and perplexity of different model sizes (169M to 14B) and data types (FP32, FP16, Q4_0, Q4_1)
2. Test on Linux (including Colab) and MacOS
3. Make required memory calculation more robust (see #4)

## How to use

### 1. Clone the repo

**Requirements**: [git](https://gitforwindows.org/).

```commandline
git clone https://github.com/saharNooby/rwkv.cpp.git
cd rwkv.cpp
```

### 2. Get the rwkv.cpp library

#### Option 2.1. Download a pre-compiled library

##### Windows

Check out [Releases](https://github.com/saharNooby/rwkv.cpp/releases), download appropriate ZIP for your CPU, extract `rwkv.dll` file into `bin\Release\` directory inside the repository directory.

To check whether your CPU supports AVX2 or AVX-512, [use CPU-Z](https://www.cpuid.com/softwares/cpu-z.html).

#### Option 2.2. Build the library yourself

##### Windows

**Requirements**: [CMake](https://cmake.org/download/), MSVC compiler.

```commandline
cmake -DBUILD_SHARED_LIBS=ON .
cmake --build . --config Release
```

If everything went OK, `bin\Release\rwkv.dll` file should appear.

### 3. Download an RWKV model from [Hugging Face](https://huggingface.co/BlinkDL) like [this one](https://huggingface.co/BlinkDL/rwkv-4-pile-169m/blob/main/RWKV-4-Pile-169M-20220807-8023.pth) and convert it into `ggml` format

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/).

```commandline
# Windows
python rwkv\convert_rwkv_to_ggml.py C:\RWKV-4b-Pile-169M-20220807-8023.pth C:\rwkv.cpp-169M.bin float32
# Linux/MacOS
python rwkv/convert_pytorch_to_ggml.py ~/Downloads/RWKV-4b-Pile-169M-20220807-8023.pth ~/Downloads/rwkv.cpp-169M.bin float32
```

#### 3.1. Optionally, quantize the model

To convert the model into INT4 quantized format, run:

```commandline
# Windows
python rwkv\quantize.py C:\rwkv.cpp-169M.bin C:\rwkv.cpp-169M-Q4_1.bin 3
# Linux / MacOS
python rwkv/quantize.py ~/Downloads/rwkv.cpp-169M.bin ~/Downloads/rwkv.cpp-169M-Q4_1.bin 3
```

Pass `2` for `Q4_0` format (smaller size, lower quality), `3` for `Q4_1` format (larger size, higher quality).

### 4. Run the model

**Requirements**: Python 3.x with [PyTorch](https://pytorch.org/get-started/locally/) and [tokenizers](https://pypi.org/project/tokenizers/).

**Note**: change the model path with the non-quantized model for the full weights model.

To generate some text, run:

```commandline
# Windows
python rwkv\generate_completions.py C:\rwkv.cpp-169M-Q4_1.bin
# Linux / MacOS
python rwkv/generate_completions.py ~/Downloads/rwkv.cpp-169M-Q4_1.bin
```

To chat with a bot, run:

```commandline
# Windows
python rwkv\chat_with_bot.py C:\rwkv.cpp-169M-Q4_1.bin
# Linux / MacOS
python rwkv/chat_with_bot.py ~/Downloads/rwkv.cpp-169M-Q4_1.bin
```

Edit [generate_completions.py](rwkv%2Fgenerate_completions.py) or [chat_with_bot.py](rwkv%2Fchat_with_bot.py) to change prompts and sampling settings.

---

Example of using `rwkv.cpp` in your custom Python script:

```python
import rwkv_cpp_model
import rwkv_cpp_shared_library

# change to model paths used above (quantized or full weights) 
model_path = r'C:\rwkv.cpp-169M.bin'


model = rwkv_cpp_model.RWKVModel(
    rwkv_cpp_shared_library.load_rwkv_shared_library(),
    model_path
)

logits, state = None, None

for token in [1, 2, 3]:
    logits, state = model.eval(token, state)
    
    print(f'Output logits: {logits}')

# Don't forget to free the memory after you've done working with the model
model.free()

```
