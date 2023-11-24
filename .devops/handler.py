import subprocess
import runpod
import os
import time

llama_cmd = os.environ.get('LLAMA_CMD', "/server --host 0.0.0.0 --threads 8 -ngl 999 -np 8 -cb -m model.gguf -c 16384")
subprocess.Popen(llama_cmd.split(' '))

## load your model(s) into vram here

def handler(event):
    print(event)
    time_slept = 0
    while time_slept < sleep_time:
        print("working, I promise")
        time_slept += 1
        time.sleep(1)
    # do the things

    return "Hello World"


runpod.serverless.start({
    "handler": handler
})
