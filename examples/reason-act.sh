#!/usr/env/bin bash

while getopts "m:" opt; do
  case $opt in
    m) MODEL="-m $OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

shift $((OPTIND-1))

cd "$(dirname "$0")"
cd ..

./bin/main $MODEL --color \
    -f ./prompts/reason-act.txt \
    -i --interactive-first \
    --top_k 10000 --temp 0.2 --repeat_penalty 1 -t 7 -c 2048 \
    -r "Question:" -r "Observation:" --in-prefix " " \
    -n -1
