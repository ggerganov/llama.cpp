
# Android

## Build on Android using Termux

[Termux](https://termux.dev/en/) is an Android terminal emulator and Linux environment app (no root required). As of writing, Termux is available experimentally in the Google Play Store; otherwise, it may be obtained directly from the project repo or on F-Droid.

With Termux, you can install and run `llama.cpp` as if the environment were Linux. Once in the Termux shell:

```
$ apt update && apt upgrade -y
$ apt install git cmake
```

Then, follow the [build instructions](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md), specifically for CMake.

Once the binaries are built, download your model of choice (e.g., from Hugging Face). It's recommended to place it in the `~/` directory for best performance:

```
$ curl -L {model-url} -o ~/{model}.gguf
```

Then, if you are not already in the repo directory, `cd` into `llama.cpp` and:

```
$ ./build/bin/llama-simple -m ~/{model}.gguf -c {context-size} -p "{your-prompt}"
```

Here, we show `llama-simple`, but any of the executables under `examples` should work, in theory. Be sure to set `context-size` to a reasonable number (say, 4096) to start with; otherwise, memory could spike and kill your terminal.

To see what it might look like visually, here's an old demo of an interactive session running on a Pixel 5 phone:

https://user-images.githubusercontent.com/271616/225014776-1d567049-ad71-4ef2-b050-55b0b3b9274c.mp4

