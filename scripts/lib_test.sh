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

pass() {
    local test_func="${FUNCNAME[1]}"
    _log 'PASSED' "$test_func"
}

fail() {
    local test_func="${FUNCNAME[1]}"
    _log 'FAILED' "$test_func: $1"
}

test_lib_sh_execution() {
    if bash lib.sh 2>/dev/null; then
        fail 'lib.sh should fail execution, but did not'
    else pass; fi
}; test_lib_sh_execution

test_isset() {
    # shellcheck disable=SC2034
    local foo=1
    if ! _isset 'foo'; then
        fail 'foo was not detecting as set'
    elif _isset 'bar'; then
        fail 'bar was detected as set'
    else pass; fi
}; test_isset

test_isnotset() {
    # shellcheck disable=SC2034
    local foo=1
    if _isnotset 'foo'; then
        fail 'foo was detected as not set'
    elif ! _isnotset 'bar'; then
        fail 'bar was detected as set'
    else pass; fi
}; test_isnotset
