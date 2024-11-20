#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
    pytest -v -s -x
else
    pytest "$@"
fi
