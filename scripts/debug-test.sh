#!/bin/bash
build_dir="build-ci-debug"
test_suite=${1:-}
test_number=${2:-}

# Function to select and debug a test
function select_test() {
    test_suite=${1:-test}
    test_number=${2:-}

    # Sanity Check If Tests Is Detected
    printf "\n\nGathering tests that fit REGEX: ${test_suite} ...\n"
    tests=($(ctest -R ${test_suite} -V -N | grep -E " +Test +#[0-9]+*" | cut -d':' -f2 | awk '{$1=$1};1'))
    if [ ${#tests[@]} -eq 0 ]
    then
        echo "No tests avaliable... check your compliation process..."
        echo "Exiting."
        exit 1
    fi

    if [ -z $test_number ]
    then
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
        read test_number
    else
        printf "\nUser Already Requested #${test_number}"
    fi

    # Start GDB with the requested test binary and arguments
    printf "Debugging(GDB) test: ${tests[test_number]}\n"
    # Change IFS (Internal Field Separator)
    sIFS=$IFS
    IFS=$'\n'

    # Get test args
    gdb_args=($(ctest -R ${test_suite} -V -N | grep "Test command" | cut -d':' -f3 | awk '{$1=$1};1' ))
    IFS=$sIFS
    printf "Debug arguments: ${gdb_args[test_number]}\n\n"

    # Expand paths if needed
    args=()
    for x in $(echo ${gdb_args[test_number]} | sed -e 's/"\/\<//' -e 's/\>"//')
    do
        args+=($(echo $x | sed -e 's/.*\/..\//..\//'))
    done

    # Execute debugger
    echo "gdb args: ${args[@]}"
    gdb --args ${args[@]}
}

# Step 0: Check the args
if [ -z "$test_suite" ] || [ "$1" = "help" ]
then
    echo "Usage: $0 [test_regex] [test_number]"
    echo "e.g., $0 test-tokenizer"
    echo "      $0 test-tokenizer 3"
    echo "Supply one regex to the script to filter tests,"
    echo "and optionally a test number to run a specific test."
    exit 1
fi

# Step 1: Reset and Setup folder context
## Sanity check that we are actually in a git repo
repo_root=$(git rev-parse --show-toplevel)
if [ ! -d "$repo_root" ]; then
    echo "Error: Not in a Git repository."
    exit 1
fi

## Reset folder to root context of git repo
pushd "$repo_root" || exit 1

## Create and enter build directory
rm -rf "$build_dir" && mkdir "$build_dir" || exit 1

# Step 2: Setup Build Environment and Compile Test Binaries
cmake -B ./build-ci-debug -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON || exit 1
pushd "$build_dir" && make -j || exit 1

# Step 3: Debug the Test
select_test "$test_suite" "$test_number"

# Step 4: Return to the directory from which the user ran the command.
popd || exit 1
popd || exit 1
popd || exit 1
