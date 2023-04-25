#!/bin/bash
set -e

MODEL=./models/ggml-vicuna-13b-1.1-q4_0.bin

# exec options
question_file=./examples/jeopardy/questions.txt
output_file=./examples/jeopardy/results.txt
opts="--temp 0 -n 80" # additional flags
prefix="Human: " # Ex. Vicuna uses "Human: "

nl='
'
introduction="You will be playing a game of Jeopardy. Simply answer the question in the correct format (Ex. What is Paris, or Who is George Washington)."

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./main -p "\"$prefix$introduction$nl$prefix$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "Current Question: $question"
  eval "$exe_cmd"
  echo -e "\n------" >> $output_file
  counter=$((counter+1))
done < "$question_file"
