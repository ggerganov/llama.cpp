#!/bin/bash

set -eu

# Start @llama.cpp scenario
behave --summary --stop --no-capture --tags llama.cpp
