# Llamacheck: Basic Spellcheck and Grammarcheck using Llama


The attached file provides a basic implementation of LLama to 
be used for Spell and Grammar checking.
We use it as follows:
```console
make llamacheck
./llamacheck <./models/llamacheck.gguf> 
```
The weights are quantized. On my machine, it runs with as speed of 7.21 t/s


Weights are available at:
https://huggingface.co/azferruolo/llamacheck