## Convert llama2.c model to ggml

This example reads weights from project [llama2.c](https://github.com/karpathy/llama2.c) and saves them in ggml compatible format.

To convert the model first download the models from the [llma2.c](https://github.com/karpathy/llama2.c) repository:

`$ make -j`

`$ ./convert-llama2c-to-ggml --vocab-model <ggml-vocab.bin> --llama2c-model <llama2.c model path> --llama2c-output-model <ggml output model path>`

Now you can use the model with command:

`$ ./main -m <ggml output model path> -p "One day, Lily met a Shoggoth" -n 500 -c 256 -eps 1e-5`
