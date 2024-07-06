
# llama-gguf-hash

CLI to hash GGUF files to detect difference on a per model and per tensor level.

**Command line options:**

- `--xxh64`: use xhash 64bit hash mode (default)
- `--sha1`: use sha1
- `--uuid`: use uuid
- `--sha256`: use sha256

## About

While most POSIX systems already have hash checking programs like sha256sum, it
is designed to check entire files. This is not ideal for our purpose if we want
to check for consistency of the tensor data even if the metadata content of the
gguf KV store has been updated.

This program is designed to hash a gguf tensor payload on a 'per tensor layer'
in addition to a 'entire tensor model' hash. The intent is that the entire
tensor layer can be checked first but if there is any detected inconsistencies,
then the per tensor hash can be used to narrow down the specific tensor layer
that has inconsistencies.

For Maintainers:
- Detection of tensor inconsistency during development and automated tests
    - This is served by xxh64 which is fast
    - This is also served by having per tensor layer to assist in narrowing down
      the location of the faulty tensor layer
    - This is also served by sha1 which is much slower but more widely supported

For Model Creators:
- Optional consistent UUID generation based on model tensor content
    - This is served by UUIDv5 which is useful for databases keys
        - llama.cpp UUIDv5 Namespace: `ef001206-dadc-5f6d-a15f-3359e577d4e5`
            - Made via UUIDv5 URL namespace of `en.wikipedia.org/wiki/Llama.cpp`

For Model Users:
- Assurance of tensor layer integrity even if metadata was updated
    - This is served by sha256 which is still considered very secure as of 2024

### Design Note

- The default behavior of this program if no arguments is provided is to hash
  using xxhash's xxh32 mode because it is very fast and is primarily targeted
  towards maintainers who may want to use this in automated tests.
- xxhash support xxh32 and xxh128 for 32bit hash and 128bit hash respectively
  however we picked 64bit xxhash as most computers are 64bit as of 2024 and thus
  would have a better affinity to calculating hash that is 64bit in size.

## Compile Example

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_FATAL_WARNINGS=ON
make -C build clean
make -C build llama-gguf-hash VERBOSE=1
./build/bin/llama-gguf-hash test.gguf
./build/bin/llama-gguf-hash --xxh64 test.gguf
./build/bin/llama-gguf-hash --sha1 test.gguf
./build/bin/llama-gguf-hash --uuid test.gguf
./build/bin/llama-gguf-hash --sha256 test.gguf
```

## Crypto/Hash Libraries Used

These micro c libraries dependencies was installed via the [clib c package manager](https://github.com/clibs)

- https://github.com/mofosyne/xxHash (From: https://github.com/Cyan4973/xxHash)
- https://github.com/clibs/sha1/
- https://github.com/jb55/sha256.c
