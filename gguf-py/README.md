## gguf

This is a Python package for writing binary files in the [GGUF](https://github.com/ggerganov/ggml/pull/302)
(GGML Universal File) format.

See [convert-llama-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-llama-hf-to-gguf.py)
as an example for its usage.

## Installation
```sh
pip install gguf
```

## Development
Maintainers who participate in development of this package are advised to install it in editable mode:

```sh
cd /path/to/llama.cpp/gguf-py

pip install --editable .
```

**Note**: This may require to upgrade your Pip installation, with a message saying that editable installation currently requires `setup.py`.
In this case, upgrade Pip to the latest:

```sh
pip install --upgrade pip
```

## Publishing
To publish the package, you need to have `twine` and `build` installed:

```sh
pip install build twine
```

Then, folow these steps to release a new version:

1. Update the version in `pyproject.toml`.
2. Build the package:

```sh
python -m build
```

3. Upload the generated distribution archives:

```sh
python -m twine upload dist/*
```

## TODO
- [ ] Add tests
- [ ] Include conversion scripts as command line entry points in this package.
- Add CI workflow for releasing the package.
