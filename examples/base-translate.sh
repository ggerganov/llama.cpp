#!/bin/bash
#
# Few-shot translation example.
# Requires a base model (i.e. no fine-tuned or instruct models).
#
# Usage:
#
#   cd llama.cpp
#   make -j
#
#   ./examples/base-translate.sh <model-base> "<text>" [extra-main-args]
#

if [ $# -lt 2 ]; then
  echo "Usage: ./base-translate.sh <model-base> \"<text>\" [extra-main-args]"
  exit 1
fi

eargs=""
if [ $# -gt 2 ]; then
  eargs="${@:3}"
fi

ftmp="__llama.cpp_example_tmp__.txt"
trap "rm -f $ftmp" EXIT

echo "Translate from English to French:

===

sea otter, peppermint, plush girafe:

sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche

===

violin

violin => violon

===

phone, computer, mouse, keyboard:

phone => téléphone
computer => ordinateur
mouse => souris
keyboard => clavier

===
" > $ftmp

echo "$2
" >> $ftmp

model=$1

# generate the most likely continuation until the string "===" is found
./llama-cli -m $model -f $ftmp -n 64 --temp 0 --repeat-penalty 1.0 --no-penalize-nl -r "===" $eargs
