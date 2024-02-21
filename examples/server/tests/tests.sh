#!/bin/bash

set -eu

# Start @llama.cpp scenario
behave --summary --stop --tags llama.cpp
