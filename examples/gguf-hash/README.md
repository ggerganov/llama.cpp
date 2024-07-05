## GGUF hash Example

CLI to hash GGUF files.

**Command line options:**

- `--xxhash`: use xhash (default)
- `--sha1`: use sha1
- `--uuid`: use uuid
- `--sha256`: use sha256

### Compile Example

```
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_FATAL_WARNINGS=ON
make -C build clean
make -C build llama-gguf-hash VERBOSE=1
./build/bin/llama-gguf-hash test.gguf
./build/bin/llama-gguf-hash --xxhash test.gguf
./build/bin/llama-gguf-hash --sha1 test.gguf
./build/bin/llama-gguf-hash --uuid test.gguf
./build/bin/llama-gguf-hash --sha256 test.gguf
```

### Crypto/Hash Libraries Used

These micro c libraries dependencies was installed via the [clib c package manager](https://github.com/clibs)

- https://github.com/mofosyne/xxHash
- https://github.com/clibs/sha1/
- https://github.com/jb55/sha256.c
