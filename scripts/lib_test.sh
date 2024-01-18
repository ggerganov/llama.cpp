#!/bin/bash
if [[ ! ${BASH_SOURCE[0]} -ef $0 ]]; then
   echo >&2 "This script should be executed, not sourced!"
   exit 1
fi
#### BEGIN SETUP #####
set -euo pipefail
this=$(realpath -- "$0"); readonly this
cd "$(dirname "$this")"
shellcheck --external-sources "$this"
# shellcheck source=lib.sh
source 'lib.sh'
#### END SETUP ####

Pass() {
    local test_func="${FUNCNAME[1]}"
    _Log 'PASSED' "$test_func"
}

Fail() {
    local test_func="${FUNCNAME[1]}"
    _Log 'FAILED' "$test_func: $1"
}

TestLibShExecution() {
    if bash lib.sh 2>/dev/null; then
        Fail 'lib.sh should fail execution, but did not'
    else Pass; fi
}; TestLibShExecution

TestIsSet() {
    # shellcheck disable=SC2034
    local foo=1
    if ! _IsSet 'foo'; then
        Fail 'foo was not detecting as set'
    elif _IsSet 'bar'; then
        Fail 'bar was detected as set'
    else Pass; fi
}; TestIsSet

TestIsNotSet() {
    # shellcheck disable=SC2034
    local foo=1
    if _IsNotSet 'foo'; then
        Fail 'foo was detected as not set'
    elif ! _IsNotSet 'bar'; then
        Fail 'bar was detected as set'
    else Pass; fi
}; TestIsNotSet
