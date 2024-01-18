#!/bin/bash

if [[ ${BASH_SOURCE[0]} -ef $0 ]]; then
   echo >&2 "This script should be sourced, not executed!"
   exit 1
fi

_Log() {
    local level=$1 msg=$2
    printf >&2 '%s: %s\n' "$level" "$msg"
}

_LogDebug() {
    _Log DEBUG "$@"
}

_LogInfo() {
    _Log INFO "$@"
}

_LogFatal() {
    _Log FATAL "$@"
    exit 1
}

# Return true if the variable with name $1 is set
_IsSet() {
    (( $# != 1 )) && return false
    if [[ -n ${!1+x} ]]; then
        return 0
    else
        return 1
    fi
}

_IsNotSet() {
    ! _IsSet "$@"
}
