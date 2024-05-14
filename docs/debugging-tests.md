# Debugging Tests Tips

## How to run & execute a specific test without anything else to keep the feedback loop short?

There is a script called run-single-test.sh in the scripts folder whose parameter takes a REGEX and an optional test number.

For example, running the following command will output an interactive list from which you can select a test. It takes this form:

`run-single-test.sh [OPTION]... <test_regex> <test_number>`

It will then build & run in the debugger for you.

```bash
./scripts/run-single-test.sh test-tokenizer
```

An example of a single test output is shown below. You will get either a green TEST PASS or a red TEST FAIL if a particular test is working or not. This shorter feedback loop will hopefully make it easier for you to figure out the problem you are trying to solve.

```bash
$ ./scripts/run-single-test.sh test 24
~/gitextern/llama.cpp ~/gitextern/llama.cpp

... prepping cmake environment ...
... building test binaries ...
... running test ...

Ran Test #24: test-eval-callback
Command: /home/mofosyne/gitextern/llama.cpp/build-ci-debug/bin/eval-callback "--hf-repo" "ggml-org/models" "--hf-file" "tinyllamas/stories260K.gguf" "--model" "stories260K.gguf" "--prompt" "hello" "--seed" "42" "-ngl" "0"
TEST PASS
```

For further reference use `run-single-test.sh -h` to print help.

### How does the script work?

This is similar to `debug-test.sh` so you can follow the similar guide in this page for similar process. Just run the command directly rather than though gdb.

## How to run & debug a specific test without anything else to keep the feedback loop short?

There is a script called debug-test.sh in the scripts folder whose parameter takes a REGEX and an optional test number.

For example, running the following command will output an interactive list from which you can select a test. It takes this form:

`debug-test.sh [OPTION]... <test_regex> <test_number>`

It will then build & run in the debugger for you.

```bash
./scripts/debug-test.sh test-tokenizer

# Once in the debugger, i.e. at the chevrons prompt, setting a breakpoint could be as follows:
>>> b main
```

For further reference use `debug-test.sh -h` to print help.

&nbsp;

### How does the script work?
If you want to be able to use the concepts contained in the script separately, the important ones are briefly outlined below.

#### Step 1: Reset and Setup folder context

From base of this repository, let's create `build-ci-debug` as our build context.

```bash
rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
```

#### Step 2: Setup Build Environment and Compile Test Binaries

Setup and trigger a build under debug mode. You may adapt the arguments as needed, but in this case these are sane defaults.

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON ..
make -j
```

#### Step 3.1: Identify Test Command for Debugging

The output of this command will give you the command & arguments needed to run GDB.

* `-R test-tokenizer` : looks for all the test files named `test-tokenizer*` (R=Regex)
* `-N` : "show-only" disables test execution & shows test commands that you can feed to GDB.
* `-V` : Verbose Mode

```bash
ctest -R "test-tokenizer" -V -N
```

This may return output similar to below (focusing on key lines to pay attention to):

```bash
...
1: Test command: ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
1: Working Directory: .
Labels: main
  Test  #1: test-tokenizer-0-llama-spm
...
4: Test command: ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-falcon.gguf"
4: Working Directory: .
Labels: main
  Test  #4: test-tokenizer-0-falcon
...
```

So for test #1 we can tell these two pieces of relevant information:
* Test Binary: `~/llama.cpp/build-ci-debug/bin/test-tokenizer-0`
* Test GGUF Model: `~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf`

#### Step 3.2: Run GDB on test command

Based on the ctest 'test command' report above we can then run a gdb session via this command below:

```bash
gdb --args ${Test Binary} ${Test GGUF Model}
```

Example:

```bash
gdb --args ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
```
