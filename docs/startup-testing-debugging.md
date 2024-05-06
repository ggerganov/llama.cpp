# Startup, Testing, & Debugging
---
I know this sort of thing can be elementary to some, but to others it can save days or weeks worth of time & because I was searching for this kind of information in the repository & couldn't seem to find it, I wanted to try to provide a brief example of getting up & running, debugging code & tests. I won't be super explicit about everything because there is documentation in the repository that can be found by searching. I'll just assume you use a terminal & a text editor because it's the simplest thing to understand.

## How to run & debug a specific test without running all the others to keep the feedback loop short?
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

# In case you're struggling after this, GDB can tell you how to use it (>>> help) & it can do incredible things like inspect the variables of any stackframe previous to the current one to help you see what's led to the issue. GDB also makes it easy to revisit a debugging session (see below).
```

## How to figure out where a bug came from?
The first issue I was assigned to was pointing me in the direction of CUDA related issues, but the root cause turned out to be tokenizer related. There is a lot going on in llama.cpp & trying to step through the entire execution & keep track of all of the many moving parts to figure out what went wrong is probably not going to work, especially if you're unfamiliar with the codebase. A better strategy might be to isolate the code change that caused the issue to appear. A great way to do that is to checkout previous tags & run the code that produces the issue until you find the first ancestor without the problem. Then you can run the debugger with no breakpoints to get a stack trace & compare the files closest to the error in the stack with the files that changed when the tag was introduced (look at tags in GitHub, they'll link you to the PR commit diff). This is similar to how I figured out what happened in my first issue.
```bash
# GDB will show you detailed stack information.
>>> i stack
```

## Debugging tricks?
After you narrow down where the problem is you might want to step through some program execution. That looks similar to what we did above, but there are some tricks that can make it easier.
```bash
# GDB can save your breakpoints to a file & source them later so that you can revisit a debugging session.
>>> save breakpoints {filename}
# In some future debugging session.
>>> source {filename}
>>> run

# GDB can show you the variables from any stackframe preceeding the one you're in currently.
>>> i stack
```

## What if my LSP is doing a poor job navigating around the repository?
Install bear (sudo apt install bear | etc..). It can watch your build system create its executables & then work out how to let your language service provider know how to navigate your project.
```bash
# This will save its output into a local file & suddenly your LSP will know how to GOTO DEFN or DECL, etc.
bear -- make -j
```
## What if I don't have a GPU locally?
I run all my code on Nvidia GPU enabled AWS Spot Instances. I use the cheapest Ubuntu based instance that gets the job done. Once I get an instance up with all the dependencies installed, I save the base work, without any credentials of course, in the form of a custom AMI which can be used to create a new EC2. You can figure out how that's done online quickly or if you have a basic understanding of EC2s you can click around the AWS console & figure it out. My rough cost per day using this workflow is ~1$. It does fluctuate with the price of whichever spot instance I'm using that day. At this point, I never even go to the AWS console, my workflow is bash scripted, but AWS makes it easy to connect to any EC2 in the AWS console & work in a terminal or in Cloud9, a browser-based IDE. I personally use SSH & NeoVim, but there's 100 different ways to get the job done. I make sure to stop all my instances each evening, but it wouldn't be too expensive if I forgot. There are some hypothetical downsides to spot instances, but I never experience them. The AWS experience has come a long way & even setting up SSO for AWSCLI authentication is relatively straight-forward now.

If you're just looking for a CPP repl experience with GPUs available, your best bet is Google Colab. Here are the relevant tools:
```bash
# Decorate a block that will represent a CPP or CUDA file like this:
%%writefile vector-math.cu || %%writefile vector-math.cpp

# In some other block, compile & run the code.
%%shell
nvcc -o vector-math vector-math.cu
./vector-math
```
This is something I do very often, it really is an incredible tool with access to high end GPUs.
