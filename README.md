# llama-for-kobold

A self contained distributable from Concedo that exposes llama.cpp function bindings, allowing it to be used via a simulated Kobold API endpoint.

![Preview](preview.png)

## Considerations
- Don't want to use pybind11 due to dependencies on MSVCC
- ZERO or MINIMAL changes as possible to main.cpp - do not move their function declarations elsewhere!
- Leave main.cpp UNTOUCHED, We want to be able to update the repo and pull any changes automatically.
- No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields. Python will ALWAYS provide the memory, we just write to it.
- No external libraries or dependencies. That means no Flask, Pybind and whatever. All You Need Is Python.

## Usage
- Windows binaries are provided in the form of **llamacpp.dll** but if you feel worried go ahead and rebuild it yourself.
- Weights are not included, you can use the llama.cpp quantize.exe to generate them from your official weight files (or download them from...places).
- To run, simply clone the repo and run `llama_for_kobold.py [ggml_quant_model.bin] [port]`, and then connect with Kobold or Kobold Lite. 
- By default, you can connect to http://localhost:5001 (you can also use https://lite.koboldai.net/?local=1&port=5001).

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, Kobold Lite is licensed under the AGPL v3.0 License
- The provided python ctypes bindings in llamacpp.dll are also under the AGPL v3.0 License

## Notes
- There is a fundamental flaw with llama.cpp, which causes generation delay to scale linearly with original prompt length. If you care, **please contribute to [this discussion](https://github.com/ggerganov/llama.cpp/discussions/229)** which, if resolved, will actually make this viable.