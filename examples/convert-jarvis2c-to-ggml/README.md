## Convert jarvis2.c model to ggml

This example reads weights from project [jarvis2.c](https://github.com/karpathy/jarvis2.c) and saves them in ggml compatible format. The vocab that is available in `models/ggml-vocab.bin` is used by default.

To convert the model first download the models from the [jarvis2.c](https://github.com/karpathy/jarvis2.c) repository:

`$ make -j`

After successful compilation, following usage options are available:
```
usage: ./jarvis-convert-jarvis2c-to-ggml [options]

options:
  -h, --help                       show this help message and exit
  --copy-vocab-from-model FNAME    path of gguf jarvis model or jarvis2.c vocabulary from which to copy vocab (default 'models/7B/ggml-model-f16.gguf')
  --jarvis2c-model FNAME            [REQUIRED] model path from which to load Karpathy's jarvis2.c model
  --jarvis2c-output-model FNAME     model path to save the converted jarvis2.c model (default ak_jarvis_model.bin')
```

An example command using a model from [karpathy/tinyjarviss](https://huggingface.co/karpathy/tinyjarviss) is as follows:

`$ ./jarvis-convert-jarvis2c-to-ggml --copy-vocab-from-model jarvis-2-7b-chat.gguf.q2_K.bin --jarvis2c-model stories42M.bin --jarvis2c-output-model stories42M.gguf.bin`

Note: The vocabulary for `stories260K.bin` should be its own tokenizer `tok512.bin` found in [karpathy/tinyjarviss/stories260K](https://huggingface.co/karpathy/tinyjarviss/tree/main/stories260K).

Now you can use the model with a command like:

`$ ./jarvis-cli -m stories42M.gguf.bin -p "One day, Lily met a Shoggoth" -n 500 -c 256`
