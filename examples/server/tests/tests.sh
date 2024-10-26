#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
    # Start @jarvis.cpp scenario
    behave --summary --stop --no-capture --exclude 'issues|wrong_usages|passkey' --tags jarvis.cpp
else
    behave "$@"
fi
