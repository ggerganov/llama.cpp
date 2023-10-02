#!/bin/bash
set -e

MODEL=./models/llama-2-7b/ggml-model-q4_0.gguf
MODEL_NAME=Llama2-7b-ggml-q4

# exec options
prefix="Question: " # Ex. Vicuna uses "Human: "
opts="--temp 0 -n 80" # additional flags
nl='
'
introduction="You will be playing a game of Jeopardy. Simply answer the question in the correct format (Ex. What is Paris, or Who is George Washington)."

# file options
question_file=./examples/jeopardy/questions.txt
# if the required model file doesn't exist, create it; otherwise update it
touch ./examples/jeopardy/results/$MODEL_NAME.txt
output_file=./examples/jeopardy/results/$MODEL_NAME.txt

counter=1

echo 'Running'
while IFS= read -r question
do
  exe_cmd="./build/bin/main -p "\"$nl$prefix$question\"" "$opts" -m ""\"$MODEL\""" >> ""\"$output_file\""
  echo $counter
  echo "Current Question: $question"
  eval "$exe_cmd"
  echo -e "\n------" >> $output_file
  counter=$((counter+1))
done < "$question_file"
