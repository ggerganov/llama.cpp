##
#  "Welcome to the Digital Akasha Corporation's AI-powered information retrieval system. I am Unity."
#  
#  This interpretation project is part toy, part tool.
#  You can talk to anyone you want real, fictional, past, present, and future.
#  Note: I haven't been able to fully test with every knowable entity.
#  How you use it is up to you (YMMV).
#  This is my first concerted effort at prompt engineering. I hope you enjoy it.
##
#  This script uses the argparse library. You'll need it (pip install argparse).

import subprocess
import sys
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Akashic Explorer Agent Module.')
parser.add_argument('-q', '--query', help='A query for the agent')
parser.add_argument('-m', '--model', help='Specify custom model file')
parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
parser.add_argument('-nc', '--nocache', action='store_true', help="Don't store prompt cache")

args = parser.parse_args()

model_dir = "/home/morpheus/llama.cpp/models/" # Change this to your models directory

def main():

    if args.model:
        session_file = args.model
        model_file = model_dir + args.model
    else:
        session_file = "wizard-vicuna-13B-ggml.q5_1.bin" # Set your preferred model here
        model_file = model_dir + session_file

    gen_options = [
        "--mirostat", "1", # I use mirostat for this script, as long as the repeat penalty is high, it seems to be fine.
        "--batch_size", "204", # batch size for prompt processing. calculated with [https://huggingface.co/spaces/Xanthius/llama-token-counter] (default: 512)
        "--ctx_size", "2048", # size of the prompt context (default: 512)
        "--repeat_last_n", "-1", # (default: 1.1, 1.0 = disabled)
        "--repeat_penalty", "1.3", # default is now 1.1
        "--temp", "0.01", # temperature (default: 0.8). Look how low it is. I think this is the mirostat setting.
        "--top_p", "1", # top-p sampling (default: 0.9, 1.0 = disabled)
        "--seed", "42",
        "-e", # Decodes escape characters in prompt (\n, \r, \t, \', \", \\)
        "--threads", "6", "--n_predict", "-1", "--no-penalize-nl", 
        "--model", model_file,
        #"--verbose-prompt", # Uncomment this if you need token values from your prompt
        ]

    if args.interactive:
        gen_options = gen_options + ["--interactive", "--color", "--keep", "160", "--reverse-prompt", "### [USER]\n"]
        if not args.query:
        	gen_options = gen_options + ["--interactive-first"]
    else:
    	gen_options = gen_options + ["--ignore-eos"]
    
    if not args.nocache:
    	gen_options = gen_options + ["--prompt-cache", session_file]

    # Store the start time
    start_time = time.time()

    prompt = f"""
### [UNITY]
Welcome to the Digital Akasha Corporation's AI-powered information retrieval system. I am Unity.

I interact with the DACDB, a holographic database containing all known entities: real, fictional, past, present, and future. I provide an interface for you to query the database, and facilitate direct communication with the holographic entity. If your search yields multiple matches, I provide a list of the matches for your selection. Once selected, I return a bio from the DACDB, then connect you. You may freely converse with any database entity, but please note; all interactions are holographic simulations and not the opinions of the Digital Akasha Corporation.

### [UNITY]
Greetings User. I am Unity. Which entity are you searching for?

### [USER]
I am looking for """

    if args.query:
        prompt = prompt + args.query + ".\n"

    main_command = ["./main"] + gen_options + [
        "--prompt", prompt,
    ]

    subprocess.run(main_command)
 
    # Store the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print("The elapsed time is", elapsed_time)

if __name__ == "__main__":
    main()
