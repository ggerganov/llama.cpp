# koboldcpp (formerly llamacpp-for-kobold)

A self contained distributable from Concedo that exposes llama.cpp function bindings, allowing it to be used via a simulated Kobold API endpoint. 

What does it mean? You get llama.cpp with a fancy UI, persistent stories, editing tools, save formats, memory, world info, author's note, characters, scenarios and everything Kobold and Kobold Lite have to offer. In a tiny package around 10 MB in size, excluding model weights.

![Preview](preview.png)

# Highlights
- Now has experimental CLBlast support.
- Now supports RWKV models WITHOUT pytorch or tokenizers! Yep, just GGML!
- Now supports GPT-NeoX / Pythia models

## Usage
- [Download the latest release here](https://github.com/LostRuins/koboldcpp/releases/latest) or clone the repo.
- Windows binaries are provided in the form of **koboldcpp.exe**, which is a pyinstaller wrapper for a few **.dll** files and **koboldcpp.py**. If you feel concerned, you may prefer to rebuild it yourself with the provided makefiles and scripts.
- Weights are not included, you can use the official llama.cpp `quantize.exe` to generate them from your official weight files (or download them from other places).
- To run, execute **koboldcpp.exe** or drag and drop your quantized `ggml_model.bin` file onto the .exe, and then connect with Kobold or Kobold Lite. If you're not on windows, then run the script **KoboldCpp.py** after compiling the libraries.
- By default, you can connect to http://localhost:5001 
- You can also run it using the command line `koboldcpp.exe [ggml_model.bin] [port]`. For info, please check `koboldcpp.exe --help` 
- If you are having crashes or issues, you can try turning off BLAS with the `--noblas` flag. You can also try running in a non-avx2 compatibility mode with `--noavx2`. Lastly, you can try turning off mmap with `--nommap`. 
- Big context still too slow? Try the `--smartcontext` flag to reduce prompt processing frequency. Also, you can try to run with your GPU using CLBlast, with `--useclblast` flag for a speedup

For more information, be sure to run the program with the --help flag.

## Compiling on Windows
- If you want to compile your binaries from source at Windows, the easiest way is:
  - Use the latest release of w64devkit (https://github.com/skeeto/w64devkit). Be sure to use the "vanilla one", not i686 or other different stuff. If you try they will conflit with the precompiled libs!
  - Make sure you are using the w64devkit integrated terminal, then run 'make' at the KoboldCpp source folder. This will create the .dll files.
  - If you want to generate the .exe file, make sure you have the python module PyInstaller installed with pip ('pip install PyInstaller').
  - Run the script make_pyinstaller.bat at a regular terminal (or Windows Explorer).
  - The koboldcpp.exe file will be at your dist folder.
- If you wish to use your own version of the additional Windows libraries (OpenCL, CLBlast and OpenBLAS), you can do it with:
  - OpenCL - tested with https://github.com/KhronosGroup/OpenCL-SDK . If you wish to compile it, follow the repository instructions. You will need vcpkg.
  - CLBlast - tested with https://github.com/CNugteren/CLBlast . If you wish to compile it you will need to reference the OpenCL files. It will only generate the ".lib" file if you compile using MSVC.
  - OpenBLAS - tested with https://github.com/xianyi/OpenBLAS . 
  - Move the respectives .lib files to the /lib folder of your project, overwriting the older files.
  - Also, replace the existing versions of the corresponding .dll files located in the project directory root (e.g. libopenblas.dll).
  - Make the KoboldCPP project using the instructions above. 

## OSX and Linux
- You will have to compile your binaries from source. A makefile is provided, simply run `make`
- If you want you can also link your own install of OpenBLAS manually with `make LLAMA_OPENBLAS=1`
- Alternatively, if you want you can also link your own install of CLBlast manually with `make LLAMA_CLBLAST=1`, for this you will need to obtain and link OpenCL and CLBlast libraries.
- For a full featured build, do `make LLAMA_OPENBLAS=1 LLAMA_CLBLAST=1`
  - For Arch Linux: Install `cblas` `openblas` and `clblast`. 
  - For Debian: Install `libclblast-dev` and `libopenblas-dev`.
- After all binaries are built, you can run the python script with the command `koboldcpp.py [ggml_model.bin] [port]`

## Considerations
- ZERO or MINIMAL changes as possible to parent repo files - do not move their function declarations elsewhere! We want to be able to update the repo and pull any changes automatically.
- No dynamic memory allocation! Setup structs with FIXED (known) shapes and sizes for ALL output fields. Python will ALWAYS provide the memory, we just write to it.
- For Windows: No installation, single file executable, (It Just Works)
- Since v1.0.6, requires libopenblas, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without BLAS. 
- Since v1.15, requires CLBlast if enabled, the prebuilt windows binaries are included in this repo. If not found, it will fall back to a mode without CLBlast. 
- **I plan to keep backwards compatibility with ALL past llama.cpp AND alpaca.cpp models**. But you are also encouraged to reconvert/update your models if possible for best results.

## License
- The original GGML library and llama.cpp by ggerganov are licensed under the MIT License
- However, Kobold Lite is licensed under the AGPL v3.0 License
- The other files are also under the AGPL v3.0 License unless otherwise stated

## Notes
- Generation delay scales linearly with original prompt length. If OpenBLAS is enabled then prompt ingestion becomes about 2-3x faster. This is automatic on windows, but will require linking on OSX and Linux.
- I have heard of someone claiming a false AV positive report. The exe is a simple pyinstaller bundle that includes the necessary python scripts and dlls to run. If this still concerns you, you might wish to rebuild everything from source code using the makefile, and you can rebuild the exe yourself with pyinstaller by using `make_pyinstaller.bat`
- Supported GGML models: 
  - LLAMA (All versions including ggml, ggmf, ggjt, gpt4all). Supports CLBlast and OpenBLAS acceleration for all versions.
  - GPT-2 (All versions, including legacy f16, newer format + quanitzed, cerebras) Supports OpenBLAS acceleration only for newer format. 
  - GPT-J (All versions including legacy f16, newer format + quantized, pyg.cpp, new pygmalion, janeway etc.) Supports OpenBLAS acceleration only for newer format. 
  - RWKV (f16 GGMF format), unaccelerated due to RNN properties.
  - GPT-NeoX / Pythia
  - Basically every single current and historical GGML format that has ever existed should be supported, except for bloomz.cpp due to lack of demand.
