#!/bin/bash

# Function to select and debug a test
function select_test() {
    test_suite="$1"

    # Sanity Check If Tests Is Detected
    printf "\n\nGathering tests that fit REGEX: ${test_suite} ...\n"
    tests=($(ctest -R ${test_suite} -V -N | grep -E " +Test +#[0-9]+*" | cut -d':' -f2 | awk '{$1=$1};1'))
    if [ ${#tests[@]} -eq 0 ]
    then
        echo "No tests avaliable ..."
        echo "Exiting."
        exit 1
    fi

    # List out avaliable tests
    printf "Which test would you like to debug?\n"
    id=0
    for s in "${tests[@]}"
    do
        echo "Test# ${id}"
        echo "  $s"
        ((id++))
    done

    # Prompt user which test they wanted to run
    printf "\nRun test#? "
    read n

    # Start GDB with the requested test binary and arguments
    printf "Debugging(GDB) test: ${tests[n]}\n"
    # Change IFS (Internal Field Separator)
    sIFS=$IFS
    IFS=$'\n'

    # Get test args
    gdb_args=($(ctest -R ${test_suite} -V -N | grep "Test command" | cut -d':' -f3 | awk '{$1=$1};1' ))
    IFS=$sIFS
    printf "Debug arguments: ${gdb_args[n]}\n\n"

    # Expand paths if needed
    args=()
    for x in $(echo ${gdb_args[n]} | sed -e 's/"\/\<//' -e 's/\>"//')
    do
        args+=($(echo $x | sed -e 's/.*\/..\//..\//'))
    done

    # Execute debugger
    gdb --args ${args[@]}
}

# Step 0: Check the args
if [ $# -ne 1 ] || [ "$1" = "help" ]
then
    echo "Supply one regex to the script, e.g. test-tokenizer would\n"
    echo "return all the tests in files that match with test-tokenizer."
    exit 1
fi

# Step 1: Prepare the Build Environment

## Sanity check that we are actually in a git repo
repo_root=$(git rev-parse --show-toplevel)
if [ ! -d "$repo_root" ]; then
    echo "Error: Not in a Git repository."
    exit 1
fi

## Build test binaries
pushd "$repo_root" || exit 1
rm -rf build-ci-debug && mkdir build-ci-debug && pushd build-ci-debug || exit 1
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON .. && popd || exit 1
make -j || exit 1
pushd tests || exit 1

# Step 2: Debug the Test
select_test "$1"

# Step 3: Return to the directory from which the user ran the command.
popd || exit 1
popd || exit 1
