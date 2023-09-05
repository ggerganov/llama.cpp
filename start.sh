#!/bin/bash

echo "Starting up"

if [ -n "$REBUILD" ]; then
  ls
  make clean 
  make -j
fi

/app/src/server "$@"
