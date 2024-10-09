#!/usr/bin/env python3
# function calling using llama-cli

import subprocess
import sys
import select
import os
import re

import json

import functions
from function_tool import get_function_tool_json, generate_schema_from_functions

function_name_list = [ name for name in dir(functions) if not name.startswith('_') ]
function_lookup = { name: getattr(functions, name) for name in function_name_list }
tools = [ get_function_tool_json(f) for (n, f) in function_lookup.items() ]
function_schema = generate_schema_from_functions(tools)

prompt = """<|start_header_id|>system<|end_header_id|>

You are capable of executing available function(s) if required.
Execute function(s) as needed.
The function calls are not shown in the conversation and should be called covertly to answer questions.
Ask for the required input to:recipient==all
Use JSON for function arguments.
Respond in this format:
>>>${recipient}
${content}
Available functions:
""" + function_schema + """<|eot_id|><|start_header_id|>system<|end_header_id|>

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files.<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

def main():
    import argparse

    parser = argparse.ArgumentParser(epilog='For more options: llama-cli --help')
    parser.add_argument('--display-prompt', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--special', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reverse-prompt', type=str, default='<|start_header_id|>user<|end_header_id|>\n')
    parser.add_argument('--ctx-size', type=int, default=1024)
    args, other_args = parser.parse_known_args()

    if args.display_prompt: print(prompt)

    command = [ './llama-cli', '-i', '-p', prompt, '--reverse-prompt', args.reverse_prompt, '--escape', '--special', '--no-display-prompt', '--log-disable', '--simple-io', '--ctx-size',  str(args.ctx_size), *other_args]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is not None: os.set_blocking(process.stdout.fileno(), False)

    try:
        run_loop(process, args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        process.terminate()
        process.wait()

def run_loop(process, args):
    pbuffer = ''
    skip_output_until_result = False
    while True:
        readable, _, _ = select.select([process.stdout, process.stderr, sys.stdin], [], [])

        for stream in readable:
            if stream == process.stdout:
                pdata = process.stdout.read()
                if not pdata: continue
                pbuffer += pdata

                if(match := re.search(r'>>>([^\n]*)\n(.*)<\|eot_id\|>', pbuffer, re.S)):
                    if not args.special:
                        pdata = pdata[:match.pos]
                    pbuffer = ''
                    skip_output_until_result = False

                    tool_name = match.group(1)
                    tool_args = match.group(2)

                    if tool_name == 'python':
                        result = functions._run_python(tool_args);
                    else:
                        try:
                            tool_args = json.loads(tool_args)
                            result = function_lookup[tool_name](**tool_args)
                        except ValueError as e:
                            result = {'error': 'unknown'}

                    result = json.dumps(result) + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
                    process.stdin.write(result + '\n')
                    process.stdin.flush()
                    if(args.special): pdata += '\n' + result
                elif (n := pdata.find('>>>')) >= 0:
                    if not args.special:
                        pdata = pdata[:n]
                        skip_output_until_result = True
                elif skip_output_until_result:
                    pdata = ''

                if not args.special:
                    pdata = re.sub(r'<\|[^\|>]*\|>', '', pdata)
                sys.stdout.write(pdata)
                sys.stdout.flush()

            elif stream == sys.stdin:
                user_input = sys.stdin.readline()
                if user_input:
                    user_input = user_input.rstrip()
                    process.stdin.write(user_input + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>' + '\n')
                    process.stdin.flush()

if __name__ == '__main__':
    main()

