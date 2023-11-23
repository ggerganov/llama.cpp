import runpod
import os
import time

sleep_time = int(os.environ.get('SLEEP_TIME', 1))

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
