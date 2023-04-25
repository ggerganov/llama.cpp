#!/bin/bash
set -e

MODEL=./models/ggml-vicuna-13b-1.1-q4_0.bin

# exec options
question_file=./examples/jeopardy/questions.txt
output_file=./examples/jeopardy/results.txt
opts="" # additional flags

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./main -p "\"$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "$question"
  eval "$exe_cmd"
  counter=$((counter+1))
done < "$question_file"
