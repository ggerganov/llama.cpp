#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
  # Start @llama.cpp scenario
  behave --summary --stop --no-capture --exclude 'issues|wrong_usages|passkey' --tags llama.cpp
else
  behave "$@"
fi

