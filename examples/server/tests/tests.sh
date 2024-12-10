#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
    pytest -v -x -m "not slow"
else
    pytest "$@"
fi
