The files in this directory give a demo of a simple Wasm program, and of trying to prove a property of division.

Standing in this directory, do

```sh
./pre-run.sh
```

This will run the `example-execution.wast` file, and perform proofs on the `div*-spec.k` files.
Two of the proofs will fail, so an error message and non-zero exit codes is to be expected.

You can then explore exection and proofs with `./kwasm klab-view [FILENAME]`.
For example, `./kwasm klab-view div3-spec.k`
See the documentation for [KLab](https://github.com/dapphub/klab) for details of how to use KLab
