#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
    pytest -v -x
else
    pytest "$@"
fi
