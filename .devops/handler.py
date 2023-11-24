import subprocess
import runpod
import os
import time
import aiohttp
import json

headers = {
    'Accept': 'text/event-stream',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Origin': 'http://127.0.0.1:8080',
    'Referer': 'http://127.0.0.1:8080/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
}

llama_cmd = os.environ.get('LLAMA_CMD', "/server --host 0.0.0.0 --threads 8 -ngl 999 -np 8 -cb -m model.gguf -c 16384")
sub = subprocess.Popen(llama_cmd.split(' '))

## load your model(s) into vram here

url = "http://0.0.0.0:8080/completion"
async def handler(event):
  print(event)
  prompt = event["input"]["prompt"]
  async with aiohttp.ClientSession() as session:
    async with session.post(url, data=json.dumps({
      'stream': True,
      'n_predict': 2048,
      'temperature': 0.2,
      'stop': [
          '</s>',
          'Llama:',
          'User:',
      ],
      'prompt': prompt,
    }), headers=headers) as response:
      async for line in response.content:
        line = line.decode('utf-8').strip()
        yield line[5:]

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True # Optional, results available via /run
})
