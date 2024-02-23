#!/bin/bash

set -eu

if [ $# -lt 1 ]
then
  # Start @llama.cpp scenario
  behave --summary --stop --no-capture --tags llama.cpp
else
  behave "$@"
fi

