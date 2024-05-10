# llama.cpp/example/embedding

This example demonstrates generate high-dimensional embedding vector of a given text with llama.cpp.

## Quick Start

To get started right away, run the following command, making sure to use the correct path for the model you have:

### Unix-based systems (Linux, macOS, etc.):

```bash
./embedding -m ./path/to/model --log-disable -p "Hello World!" 2>/dev/null
```

### Windows:

```powershell
embedding.exe -m ./path/to/model --log-disable -p "Hello World!" 2>$null
```

The above command will output space-separated float values.
