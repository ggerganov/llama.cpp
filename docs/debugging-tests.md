# How to run & debug a specific test without anything else to keep the feedback loop short?
Borrowing from the CI scripts to make something workflow specific, we have the following.
```bash
rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON .. && cd ..
# runs faster
make -j
# if you see the "you don't have "cache" installed warning, install it to save immense amounts of time!"
cd tests

# The output of this command will give you the command & arguments needed to run GDB.
# -R test-tokenizer -> looks for all the test files named test-tokenizer* (R=Regex)
# -N -> "show-only" disables test execution & shows test commands that you can feed to GDB.
ctest -R test-tokenizer-0 -V -N
# OUPUT:
    # 1: Test command: /home/ubuntu/workspace/llama.cpp/bin/test-tokenizer-0 "/home/ubuntu/workspace/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
    # 1: Working Directory: .
    # Labels: main
    # Test #1: test-tokenizer-0-llama-spm

# Now that we have the command & arguments needed to run a test, we can debug it with GDB.
# I copied the command from the above output, your scenario will be different.
gdb --args /home/ubuntu/workspace/llama.cpp/bin/test-tokenizer-0 "/home/ubuntu/workspace/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
```
