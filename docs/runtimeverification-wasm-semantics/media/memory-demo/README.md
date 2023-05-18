Building and running example

```
kompile --syntax-module WASM wasm.k
krun example-memory.wast
```

Proving

```
kompile --syntax-module WASM --backend java wasm.k
kprove memory-spec.k
```
