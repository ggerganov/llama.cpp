#!/bin/bash
#
# check-requirements.sh checks all requirements files for each top-level
# convert*.py script.
#
# WARNING: This is quite IO intensive, because a fresh venv is set up for every
# python script. As of 2023-12-22, this writes ~2.7GB of data. An adequately
# sized tmpfs /tmp or ramdisk is recommended if running this frequently.
#
# usage:    ./check-requirements.sh [<working_dir>]
#           ./check-requirements.sh 'nocleanup' [<working_dir>]
#
# where:
#           - <working_dir> is a directory that can be used as the base for
#               setting up the venvs. Defaults to `/tmp`.
#           - 'nocleanup' as the first argument will disable automatic cleanup
#               of the files created by this script.
#
# requires:
#           - bash >= 3.2.57
#           - shellcheck
#
# For each script, it creates a fresh venv, `pip install -r` the
# requirements, and finally executes the python script with no arguments to
# check for a `ModuleNotFoundError`.
#

log() {
    local level="$1"; shift
    local format="$1"; shift
    # shellcheck disable=SC2059
    >&2 printf "$level: $format\n" "$@"
}

debug () {
    log 'DEBUG' "$@"
}

info() {
    log 'INFO' "$@"
}

fatal() {
    log 'FATAL' "$@"
    exit 1
}

cleanup() {
    if [[ -n ${workdir+x} && -d $workdir && -w $workdir ]]; then
        info "Removing $workdir"
        (
            count=0
            rm -rfv "$workdir" | while read -r; do
                if (( count++ > 750 )); then
                    printf '.'
                    count=0
                fi
            done
            printf '\n'
        )&
        wait $!
        info "Removed '$workdir'"
    fi
}

abort() {
    cleanup
    exit 1
}

if [[ $1 == nocleanup ]]; then
    shift # discard nocleanup arg
else
    trap abort SIGINT SIGTERM SIGQUIT SIGABRT
    trap cleanup EXIT
fi

set -eu -o pipefail
this="$(realpath "$0")"; readonly this
cd "$(dirname "$this")"

shellcheck "$this"

readonly reqs_dir='./requirements'

workdir=
if [[ -n ${1+x} ]]; then
    arg_dir="$(realpath "$1")"
    if [[ ! ( -d $arg_dir && -w $arg_dir ) ]]; then
        fatal "$arg_dir is not a valid directory"
    fi
    workdir="$(mktemp -d "$arg_dir/check-requirements.XXXX")"
else
    workdir="$(mktemp -d "/tmp/check-requirements.XXXX")"
fi
readonly workdir

info "Working directory: $workdir"

assert_arg_count() {
    local argcount="$1"; shift
    if (( $# != argcount )); then
        fatal "${FUNCNAME[1]}: incorrect number of args"
    fi
}

check_requirements() {
    assert_arg_count 2 "$@"
    local venv="$1"
    local reqs="$2"

    info "$reqs: beginning check"
    (
        # shellcheck source=/dev/null
        source "$venv/bin/activate"
        pip --disable-pip-version-check install -q -r "$reqs"
    )
    info "$reqs: OK"
}

check_convert_script() {
    assert_arg_count 1 "$@"
    local py="$1"; shift                     # e.g. ./convert-hf-to-gguf.py
    local pyname; pyname="$(basename "$py")" # e.g. convert-hf-to-gguf.py
    pyname="${pyname%.py}"                   # e.g. convert-hf-to-gguf

    info "$py: beginning check"

    local reqs="$reqs_dir/requirements-$pyname.txt"
    if [[ ! -r "$reqs" ]]; then
        fatal "$py missing requirements. Expected: $reqs"
    fi

    local venv="$workdir/$pyname-venv"
    python3 -m venv "$venv"

    check_requirements "$venv" "$reqs"

    # Because we mask the return value of the subshell,
    # we don't need to use set +e/-e.
    # shellcheck disable=SC2155
    local py_err=$(
        # shellcheck source=/dev/null
        source "$venv/bin/activate"
        python "$py" 2>&1
    )

    # shellcheck disable=SC2181
    if grep -Fe 'ModuleNotFoundError' <<< "$py_err"; then
        fatal "$py: some imports not declared in $reqs"
    fi
    info "$py: imports OK"
}

readonly ignore_eq_eq='check_requirements: ignore "=="'

for req in "$reqs_dir"/*; do
    # Check that all sub-requirements are added to top-level requirements.txt
    if ! grep -qFe "$req" ./requirements.txt; then
        fatal "$req needs to be added to ./requirements.txt"
    fi

    # Make sure exact release versions aren't being pinned in the requirements
    # Filters out the ignore string
    req_no_ignore_eq_eq="$(grep -vF "$ignore_eq_eq" "$req")"
    if grep -Fe '==' <<< "$req_no_ignore_eq_eq" ; then
        fatal "Avoid pinning exact package versions. Use '~=' instead.\nYou can suppress this error by appending the following to the line: \n\t# $ignore_eq_eq"
    fi
done

all_venv="$workdir/all-venv"
python3 -m venv "$all_venv"
check_requirements "$all_venv" './requirements.txt'

check_convert_script './convert.py'
for py in ./convert-*.py;do
    check_convert_script "$py"
done

info "Done! No issues found."
