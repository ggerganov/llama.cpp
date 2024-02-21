#!/bin/bash

# kill any dandling server at the end
cleanup() {
    pkill -P $$
}
trap cleanup EXIT

set -eu

# Start @llama.cpp scenario
behave --summary --stop --tags llama.cpp
