## gguf

This is a Python package for writing binary files in the [GGUF](https://github.com/ggerganov/ggml/pull/302)
(GGML Universal File) format.

See [convert-llama-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-llama-hf-to-gguf.py)
as an example for its usage.

## Install
```sh
pip install gguf
```

## Development
Maintainers who participate in development of this package are advised to install it in editable mode:


```sh
cd /path/to/llama.cpp/gguf

pip install --editable .
```

**Note**: This may require to upgrade your Pip installation, with a message saying that editable installation currently requires `setup.py`.
In this case, upgrade Pip to the latest:

```sh
pip install --upgrade pip
```

## TODO

- [ ] Add tests
- [ ] Include conversion scripts as command line entry points in this package.
