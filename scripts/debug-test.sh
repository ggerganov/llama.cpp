#!/bin/bash

function select_test() {
    test_suite="$1"
    printf "\n\nGathering tests that fit the REGEX: ${test_suite} ...\n"
    printf "Which test would you like to debug?\n"
    id=0
    tests=($(ctest -R ${test_suite} -V -N | grep "Test.\ #*" | cut -d':' -f2 | awk '{$1=$1};1'))
    gdb_params=($(ctest -R test-tokenizer -V -N | grep "Test command" | cut -d':' -f3 | awk '{$1=$1};1'))
    for s in "${tests[@]}"
    do
        echo "Test# ${id}"
        echo "  $s"
        ((id++))
    done

    printf "\nRun test#? "
    read n

    printf "Debugging(GDB) test: ${tests[n]} ...\n\n"
    test=${gdb_params[n*2]}
    test_arg=$(echo ${gdb_params[n*2+1]} | sed -e 's/^.//' -e 's/.$//')
    gdb --args ${test} ${test_arg}
}

# Step 0: Check the args
if [ $# -ne 1 ] || [ $1 == "help"]
then
    echo "Supply one regex to the script, e.g. test-tokenizer would\n"
    echo "return all the tests in files that match with test-tokenizer."
    exit 1
fi




# Step 1: Prepare the Build Environment
pushd $(git rev-parse --show-toplevel)
rm -rf build-ci-debug && mkdir build-ci-debug && pushd build-ci-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON .. && popd
make -j
pushd tests

# Step 2: Debug the Test
select_test $1

# Step 3: Return to the directory from which the user ran the command.
popd
popd
