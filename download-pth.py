import os
import sys
from tqdm import tqdm
import requests

if len(sys.argv) < 3:
    print("Usage: download-pth.py dir-model model-type\n")
    print("  model-type: Available models 7B, 13B, 30B or 65B")
    sys.exit(1)

modelsDir = sys.argv[1]
model = sys.argv[2]

resolved_path = os.path.abspath(os.path.join(modelsDir, model))
os.makedirs(resolved_path, exist_ok=True)

if model == 'alpaca':
    dest_path = os.path.abspath(os.path.join(resolved_path, "ggml-alpaca-7b-q4.bin"))
    if os.path.exists(dest_path):
        print(f"Skip file download, it already exists: {dest_path}")
        sys.exit(1)

    response = requests.get("https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC", stream=True)
    with open(dest_path, 'wb') as f:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc="ggml-alpaca-7b-q4.bin") as t:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    t.update(len(chunk))
    
    sys.exit(0)


num = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}

if model not in num:
    print(f"Error: model {model} is not valid, provide 7B, 13B, 30B, 65B or alpaca")
    sys.exit(1)

print(f"Downloading model {model}")

files = ["checklist.chk", "params.json"]

for i in range(num[model]):
    files.append(f"consolidated.0{i}.pth")


for file in files:
    dest_path = os.path.join(resolved_path, file)
    
    if os.path.exists(dest_path):
        print(f"Skip file download, it already exists: {file}")
        continue

    url = f"https://agi.gpt4.org/llama/LLaMA/{model}/{file}"
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=file) as t:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    t.update(len(chunk))

files2 = ["tokenizer_checklist.chk", "tokenizer.model"]
for file in files2:
    dest_path = os.path.join(modelsDir, file)
    
    if os.path.exists(dest_path):
        print(f"Skip file download, it already exists: {file}")
        continue
    
    url = f"https://agi.gpt4.org/llama/LLaMA/{file}"
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=file) as t:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    t.update(len(chunk))