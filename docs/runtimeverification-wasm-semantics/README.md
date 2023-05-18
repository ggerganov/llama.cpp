KWasm: Semantics of WebAssembly in K
====================================

### Want to Support KWasm Development?
[Contribute to our Gitcoin Grant.](https://gitcoin.co/grants/592/kewasm-and-kwasm)

---

This repository presents a prototype formal semantics of [WebAssembly].
It is currently under construction.
For examples of current capabilities, see the unit tests under the `tests/simple` directory.

Repository Structure
--------------------

### Semantics Layout

The following files constitute the KWasm semantics:

-   [data.md](data.md) provides the (functional) data of WebAssembly (basic types, type constructors, and values).
-   [numeric.md](numeric.md) provides the functional rules for numeric operators.
-   [wasm.md](wasm.md) is the main KWasm semantics, containing the configuration and transition rules of WebAssembly.

These additional files extend the semantics to make the repository more useful:

-   [test.md](test.md) is an execution harness for KWasm, providing a simple language for describing tests/programs.

### Example usage: `./kwasm` runner script

After building the definition, you can run the definition using `./kwasm`.
The most up-to-date documentation will always be in `./kwasm help`.

Run the file `tests/simple/arithmetic.wast`:

```sh
./kwasm run tests/simple/arithmetic.wast
```

To run proofs, you can similarly use `./kwasm`, but must specify the module to use for proving.
For example, to prove the specification `tests/proofs/simple-arithmetic-spec.k`:

```sh
./kwasm prove tests/proofs/simple-arithmetic-spec.k KWASM-LEMMAS
```

You can optionally override the default backend using the `--backend BACKEND` flag:

```sh
./kwasm run   --backend llvm    tests/simple/arithmetic.wast
./kwasm prove --backend haskell tests/proofs/simple-arithmetic-spec.k KWASM-LEMMAS
```

Installing/Building
-------------------

### K Backends

There are two backends of K available, the LLVM backend for concrete execution, and the Haskell backend for symbolic reasoning and proofs.
This repository generates the build-products for the backends in `.build/defn/llvm` and `.build/defn/haskell`.

### Dependencies

The following are needed for building/running KWasm:

-   [git](https://git-scm.com/)
-   [Pandoc >= 1.17](https://pandoc.org) is used to generate the `*.k` files from the `*.md` files.
-   GNU [Bison](https://www.gnu.org/software/bison/), [Flex](https://github.com/westes/flex), and [Autoconf](http://www.gnu.org/software/autoconf/).
-   GNU [libmpfr](http://www.mpfr.org/) and [libtool](https://www.gnu.org/software/libtool/).
-   Java 8 JDK (eg. [OpenJDK](http://openjdk.java.net/))
-   [Haskell Stack](https://docs.haskellstack.org/en/stable/install_and_upgrade/#installupgrade).
    Note that the version of the `stack` tool provided by your package manager might not be recent enough.
    Please follow installation instructions from the Haskell Stack website linked above.
-   [LLVM](https://llvm.org/) For building the LLVM backend.
-   [Z3](https://github.com/Z3Prover/z3) version 4.8.15

On Ubuntu >= 15.04 (for example):

```sh
sudo apt-get install --yes                                                            \
    autoconf bison clang++-8 clang-8 cmake curl flex gcc libboost-test-dev libffi-dev \
    libgmp-dev libjemalloc-dev libmpfr-dev libprocps-dev libprotobuf-dev libtool      \
    libyaml-dev lld-8 llvm llvm-8 llvm-8-tools maven openjdk-8-jdk pandoc pkg-config  \
    protobuf-compiler python3 python-pygments python-recommonmark python-sphinx time  \
    zlib1g-dev
```

To upgrade `stack` (if needed):

```sh
stack upgrade
export PATH=$HOME/.local/bin:$PATH
```

After installing the above dependencies, make sure the submodules are setup:

```sh
git submodule update --init --recursive
```

Install repository specific dependencies:

```sh
make deps
```

#### Installing Z3

Note that KWASM requires Z3 version 4.8.15, which you may need to install from a
source build if your package manager supplies a different version. To do so,
follow the instructions
[here](https://github.com/Z3Prover/z3#building-z3-using-make-and-gccclang) after
checking out the correct tag in the Z3 repository:

```sh
git clone https://github.com/Z3Prover/z3.git
cd z3
git checkout z3-4.8.15
python scripts/mk_make.py
cd build
make
sudo make install
```

### Building

And then build the semantics:

```sh
make build
```

To only build a specific backend, you can do `make build-llvm` or `make build-haskell`.

### Media and documents

The `media/` directory contains presentations and reports about about KWasm.
The documents are named with an approximate date of presentation/submission, what type of document it is, and a brief contextual name, e.g., name of conference where it was held.

[GhostScript](https://www.ghostscript.com/) is a dependency for building documents of type `report`.

```sh
sudo apt install ghostscript
```

To build all documents in the media file:

```sh
make media
```

Testing
-------

The target `test` contains all the currently passing tests.

```sh
make test
```

Resources
---------

[WebAssembly]: <https://webassembly.github.io/spec/>
