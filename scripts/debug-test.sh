#!/bin/bash
test_suite=${1:-}
test_number=${2:-}

PROG=${0##*/}
build_dir="build-ci-debug"

# Print Color Commands
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
blue=$(tput setaf 4)
normal=$(tput sgr0)

print_full_help() {
  cat << EOF
Usage: $PROG [OPTION]... <test_regex> (test_number)
Debug specific ctest program.

Options:
  -h, --help            display this help and exit
  -g                    run in gdb mode

Arguments:
  <test_regex>     (Mandatory) Supply one regex to the script to filter tests
  (test_number)    (Optional) Test number to run a specific test

Example:
  $PROG test-tokenizer
  $PROG test-tokenizer 3
EOF
}

abort() {
  echo "Error: $1" >&2
  cat << EOF >&2
Usage: $PROG [OPTION]... <test_regex> (test_number)
Debug specific ctest program.
Refer to --help for full instructions.
EOF
  exit 1
}

if [ x"$1" = x"-h" ] || [ x"$1" = x"--help" ]; then
  print_full_help >&2
  exit 0
fi

# Parse command-line options
gdb_mode=false
while getopts "hg" opt; do
    case $opt in
        h)
            print_full_help >&2
            exit 0
            ;;
        g)
            gdb_mode=true
            echo "gdb_mode Mode Enabled"
            ;;
    esac
done

# Function to select and debug a test
function select_test() {
    test_suite=${1:-test}
    test_number=${2:-}
    repo_root=${3:-}

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
        printf "\nUser Already Requested #${test_number}\n"
    fi

    # Start GDB with the requested test binary and arguments
    printf "Debugging(GDB) test: ${tests[test_number]}\n"
    # Change IFS (Internal Field Separator)
    sIFS=$IFS
    IFS=$'\n'

    # Get test args
    test_args=($(ctest -R ${test_suite} -V -N | grep "Test command" | cut -d':' -f3 | awk '{$1=$1};1' ))
    IFS=$sIFS

    printf "${blue}Running Test #${test_number}: ${tests[test_number]}${normal}\n"
    printf "${blue}test_args[test_number]: ${test_args[test_number]}${normal}\n"

    if [ "$gdb_mode" = "true" ]; then
        # Expand paths if needed
        gdb_args=()
        for x in $(echo ${test_args[test_number]} | sed -e 's/"\/\<//' -e 's/\>"//')
        do
            gdb_args+=($(echo $x | sed -e 's/.*\/..\//..\//'))
        done

        printf "${blue}gdb_args @: ${gdb_args[@]}${normal}\n"

        # Execute debugger
        pushd "$repo_root" || exit 1
        gdb --args ${gdb_args[@]}
        popd > /dev/null || exit 1
    else

        # Execute Test
        pushd "$repo_root" || exit 1
        eval "${test_args[test_number]}"
        exit_code=$?
        popd > /dev/null || exit 1

        # Print Result
        printf "${blue}Ran Test #${test_number}: ${tests[test_number]}${normal}\n"
        printf "${yellow}Command: ${test_args[test_number]}${normal}\n"
        if [ $exit_code -eq 0 ]; then
            printf "${green}TEST PASS${normal}\n"
        else
            printf "${red}TEST FAIL${normal}\n"
        fi
    fi
}

# Step 0: Check the args
if [ -z "$test_suite" ]
then
    echo "Usage: $PROG [OPTION]... <test_regex> (test_number)"
    echo "Supply one regex to the script to filter tests,"
    echo "and optionally a test number to run a specific test."
    echo "Use --help flag for full instructions"
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
# Note: test-eval-callback requires -DLLAMA_CURL
cmake -B "./$build_dir" -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON -DLLAMA_CURL=1 || exit 1
pushd "$build_dir" && make -j || exit 1

# Step 3: Debug the Test
select_test "$test_suite" "$test_number" "$repo_root"

# Step 4: Return to the directory from which the user ran the command.
popd > /dev/null || exit 1
popd > /dev/null || exit 1
