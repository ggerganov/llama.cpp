Binary Parser for Wasm Modules
==============================

This python module converts a binary Wasm module into the Kast format accepted by KWasm.


Usage:
------

The entry point is the `wasm2kast` module.
Ensure you have `pyk` installed and available in your Python path.

Import `wasm2kast` in your Python-script.
Pass the module as bytes to the `wasm2kast` function.
It will return the Kast representation in Wasm for that module.

Example:

```py
import wasm2kast

filename = 'hello.wasm'
with open(filename, 'rb') as f:
    kast = wasm2kast.wasm2kast(f)
    print(kast)
```
