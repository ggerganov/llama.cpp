
# Example

The FIM (Fill-In-Middle) objective is useful for generating text conditioned on a prefix and a suffix.
This example is for use with codellama, for doing exactly that.

For a quick summary of what's going on here, see issue #2818, and/or read [the FIM paper](https://arxiv.org/abs/2207.14255).

```
Usage: ./fill-in-middle <model> <prefix> <suffix> <n_max_tokens> <n_threads>
```
```sh
./fill-in-middle \
   CodeLlama-34B-GGUF/codellama-34b.Q4_K_S.gguf \
   $'def add(a, b):\n' \
   $'\n' \
   64 \
   4
```

With prefix:
```py
def add(a, b):

```

And a newline as suffix:
```py

```

We can expect it to generate somethng like:
```py
    return a + b
```
