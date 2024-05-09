# How to run & debug a specific test without anything else to keep the feedback loop short?
There is a script called debug-test.sh in the scripts folder that takes a REGEX as its only parameter. For example, running the following command will output an interactive list from which you can select a test. It will then build & run in the debugger for you.
```bash
./scripts/debug-test.sh test-tokenizer

# Once in the debugger, i.e. at the chevrons prompt, setting a breakpoint could be as follows:
>>> b main
```

&nbsp;

# How does the script work?
If you want to be able to use the concepts contained in the script separately, the important ones are briefly outlined below.
```bash
# Step 1: Prepare the Build Environment
rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON .. && cd ..

# Step 2: Build the Project
make -j

# Step 3: Navigate to Test Directory
# Tip: If you see a warning about missing "cache" during installation. Then install it to save immense amounts of time!
cd tests

# Step 4: Identify Test Command for Debugging
# The output of this command will give you the command & arguments needed to run GDB.
# -R test-tokenizer : looks for all the test files named test-tokenizer* (R=Regex)
# -N : "show-only" disables test execution & shows test commands that you can feed to GDB.
# -V : Verbose Mode
ctest -R ${test_suite} -V -N

# Step 5: Debug the Test
gdb --args ${test_path} ${test_args}
gdb --args /home/ubuntu/workspace/llama.cpp/bin/test-tokenizer-0 "/home/ubuntu/workspace/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"


```



