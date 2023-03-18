#!/usr/bin/env python3

import subprocess
import pathlib
import io
import argparse

MODELS_DIR = pathlib.Path(__file__).resolve().parent.joinpath("models")
PROMPTS_DIR = pathlib.Path(__file__).resolve().parent.joinpath("prompts")

ALPACA_MODEL_FILE = MODELS_DIR.joinpath('7b/ggml-alpaca-7b-q4.bin')

SYSTEM_INP = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
PROMPT_TEMPL = "### Instruction:\n\n"
RESPONSE_TEMPL = "### Response:\n\n"

def read_prompt_file(prompt_name):
  prompt_file = PROMPTS_DIR.joinpath(f"{prompt_name}.txt")

  if not prompt_file.is_file():
      print(f"Error: {prompt_name}.txt not found in the prompts directory.")
      return None

  with open(prompt_file, 'r') as file:
      content = file.read()

  return content

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--prompt-file", 
    help="Name of the prompt text file without the .txt extension", required=False, 
    default='default')

  args = parser.parse_args()
  prompt_file_name = args.prompt_file

  prompt = PROMPT_TEMPL + read_prompt_file(prompt_file_name) + "\n\n"
  prompt = SYSTEM_INP + prompt + RESPONSE_TEMPL

  process = subprocess.Popen(['./main',
                  '-m', ALPACA_MODEL_FILE,
                  '--temp', '0.1',
                  '--top_p', '0.95',
                  '-p', prompt],
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE)
  with process.stdout:
    reader = io.TextIOWrapper(process.stdout, encoding='utf8')
    while True:
      if process.poll() is not None:
          break
      char = reader.read(1)
      print(char, end="", flush=True)

if __name__ == '__main__':
  main()
