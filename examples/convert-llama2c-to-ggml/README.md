## Convert llama2.c model to ggml

This example reads weights from project [llama2.c](https://github.com/karpathy/llama2.c) and saves them in ggml compatible format. The vocab that is available in `models/ggml-vocab.bin` is used by default.

To convert the model first download the models from the [llma2.c](https://github.com/karpathy/llama2.c) repository:

`$ make -j`

After successful compilation, following usage options are available:
```
usage: ./convert-llama2c-to-ggml [options]

options:
  -h, --help                       show this help message and exit
  --copy-vocab-from-model FNAME    model path from which to copy vocab (default 'tokenizer.bin')
  --llama2c-model FNAME            [REQUIRED] model path from which to load Karpathy's llama2.c model
  --llama2c-output-model FNAME     model path to save the converted llama2.c model (default ak_llama_model.bin')
```

An example command using a model from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) is as follows:

`$ ./convert-llama2c-to-ggml --copy-vocab-from-model ../llama2.c/tokenizer.bin --llama2c-model stories42M.bin --llama2c-output-model stories42M.ggmlv3.bin`

For now the generated model is in the legacy GGJTv3 format, so you need to convert it to gguf manually:

`$ python ./convert-llama-ggmlv3-to-gguf.py --eps 1e-5 --input stories42M.ggmlv3.bin --output stories42M.gguf.bin`

Now you can use the model with a command like:

`$ ./main -m stories42M.gguf.bin -p "One day, Lily met a Shoggoth" -n 500 -c 256`
