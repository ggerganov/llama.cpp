#!/bin/bash
#
# check-requirements.sh checks all requirements files for each top-level
# convert*.py script.
#
# WARNING: This is quite IO intensive, because a fresh venv is set up for every
# python script.
#
# requires:
# * bash >= 3.2.57
# * shellcheck
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

trap abort SIGINT SIGTERM SIGQUIT SIGABRT
trap cleanup EXIT

set -eu -o pipefail
this="$(realpath "$0")"
readonly this
cd "$(dirname "$this")"

shellcheck "$this"

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
    local py="$1"
    local pyname="${py%.py}"

    info "$py: beginning check"

    local reqs="requirements-$pyname.txt"
    local venv="$workdir/$pyname-venv"
    python3 -m venv "$venv"

    check_requirements "$venv" "$reqs"
    set +e
    (
        # shellcheck source=/dev/null
        source "$venv/bin/activate"
        py_err="$workdir/$pyname.out"
        python "$py" 2> "$py_err"
        >&2 cat "$py_err"
        grep -e 'ModuleNotFoundError' "$py_err"
    )
    set -e
    # shellcheck disable=SC2181
    (( $? )) && fatal "$py: some imports not declared in $reqs"
    info "$py: imports OK"
}

# Check requirements.txt
all_venv="$workdir/all-venv"
python3 -m venv "$all_venv"
check_requirements "$all_venv" 'requirements.txt'

check_convert_script 'convert.py'
for py in convert-*.py; do
    check_convert_script "$py"
done

info "Done! No issues found."
