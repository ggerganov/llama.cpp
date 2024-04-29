# This script downloads the tokenizer models of the specified models from Huggingface and
# generates the get_vocab_base_pre() function for convert-hf-to-gguf.py
#
# This is necessary in order to analyze the type of pre-tokenizer used by the model and
# provide the necessary information to llama.cpp via the GGUF header in order to implement
# the same pre-tokenizer.
#
# ref: https://github.com/ggerganov/llama.cpp/pull/6920
#
# Instructions:
#
# - Add a new model to the "models" list
# - Run the script with your huggingface token:
#
#   python3 convert-hf-to-gguf-update.py <huggingface_token>
#
# - Copy-paste the generated get_vocab_base_pre() function into convert-hf-to-gguf.py
# - Update llama.cpp with the new pre-tokenizer if necessary
#
# TODO: generate tokenizer tests for llama.cpp
# TODO: automate the update of convert-hf-to-gguf.py
#

import os
import requests
import sys
import json

from hashlib import sha256
from enum import IntEnum, auto

class TOKENIZER_TYPE(IntEnum):
    SPM = auto()
    BPE = auto()
    WPM = auto()

# TODO: this string has to exercise as much pre-tokenizer functionality as possible
#       will be updated with time - contributions welcome
chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````\"\"\"\"......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

if len(sys.argv) == 2:
    token = sys.argv[1]
else:
    print("Usage: python convert-hf-to-gguf-update.py <huggingface_token>")
    sys.exit(1)

# TODO: add models here, base models preferred
models = [
        { "name": "llama-spm",      "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/meta-llama/Llama-2-7b-hf", },
        { "name": "llama-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/meta-llama/Meta-Llama-3-8B", },
        { "name": "phi-3",          "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct", },
        { "name": "deepseek-llm",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-base", },
        { "name": "deepseek-coder", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base", },
        { "name": "falcon",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/falcon-7b", },
        { "name": "bert-bge",       "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/BAAI/bge-small-en-v1.5", },
        { "name": "mpt",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mosaicml/mpt-7b", },
        { "name": "starcoder",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigcode/starcoder2-3b", },
        { "name": "gpt-2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/openai-community/gpt2", },
        ]

# make directory "models/tokenizers" if it doesn't exist
if not os.path.exists("models/tokenizers"):
    os.makedirs("models/tokenizers")

def download_file_with_auth(url, token, save_path):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File {save_path} downloaded successfully")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# download the tokenizer models
for model in models:
    name = model["name"]
    repo = model["repo"]
    tokt = model["tokt"]

    if not os.path.exists(f"models/tokenizers/{name}"):
        os.makedirs(f"models/tokenizers/{name}")
    else:
        print(f"Directory models/tokenizers/{name} already exists - skipping")
        continue

    print(f"Downloading {name} to models/tokenizers/{name}")

    url = f"{repo}/raw/main/config.json"
    save_path = f"models/tokenizers/{name}/config.json"
    download_file_with_auth(url, token, save_path)

    url = f"{repo}/raw/main/tokenizer.json"
    save_path = f"models/tokenizers/{name}/tokenizer.json"
    download_file_with_auth(url, token, save_path)

    if tokt == TOKENIZER_TYPE.SPM:
        url = f"{repo}/resolve/main/tokenizer.model"
        save_path = f"models/tokenizers/{name}/tokenizer.model"
        download_file_with_auth(url, token, save_path)

    url = f"{repo}/raw/main/tokenizer_config.json"
    save_path = f"models/tokenizers/{name}/tokenizer_config.json"
    download_file_with_auth(url, token, save_path)

# generate the source code for the convert-hf-to-gguf.py:get_vocab_base_pre() function:
# TODO: auto-update convert-hf-to-gguf.py with the generated function

src_ifs = ""
for model in models:
    name = model["name"]
    tokt = model["tokt"]

    if tokt == TOKENIZER_TYPE.SPM:
        continue

    # create the tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")

    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()

    print(f"model: {name}")
    print(f"tokt: {tokt}")
    print(f"repo: {model['repo']}")
    print(f"chktok: {chktok}")
    print(f"chkhsh: {chkhsh}")

    # print the "pre_tokenizer" content from the tokenizer.json
    with open(f"models/tokenizers/{name}/tokenizer.json", "r") as f:
        cfg = json.load(f)
        pre_tokenizer = cfg["pre_tokenizer"]
        print("pre_tokenizer: " + json.dumps(pre_tokenizer, indent=4))

    print(f"\n")

    src_ifs += f"        if chkhsh == \"{chkhsh}\":\n"
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f"            res = \"{name}\"\n"

src_func = ""
src_func +=  "    def get_vocab_base_pre(self, tokenizer) -> str:\n"
src_func +=  "        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that\n"
src_func +=  "        # is specific for the BPE pre-tokenizer used by the model\n"
src_func +=  "        # we will use this unique identifier to write a \"tokenizer.ggml.pre\" entry in the GGUF file which we can\n"
src_func +=  "        # use in llama.cpp to implement the same pre-tokenizer\n"
src_func +=  "\n"
src_func += f"        chktxt = {repr(chktxt)}\n"
src_func +=  "\n"
src_func +=  "        chktok = tokenizer.encode(chktxt)\n"
src_func +=  "        chkhsh = sha256(str(chktok).encode()).hexdigest()\n"
src_func +=  "\n"
src_func +=  "        print(f\"chktok: {chktok}\")\n"
src_func +=  "        print(f\"chkhsh: {chkhsh}\")\n"
src_func +=  "\n"
src_func +=  "        res = None\n"
src_func +=  "\n"
src_func +=  "        # NOTE: if you get an error here, you need to add the model to the if-elif chain below\n"
src_func +=  "        #       don't do this manually - use the convert-hf-to-gguf-update.py script!\n"
src_func += f"{src_ifs}\n"
src_func +=  "        if res is None:\n"
src_func +=  "            print(\"\\n\")\n"
src_func +=  "            print(\"**************************************************************************************\")\n"
src_func +=  "            print(\"** WARNING: The BPE pre-tokenizer was not recognized!\")\n"
src_func +=  "            print(\"**          This means that it was not added yet or you are using an older version.\")\n"
src_func +=  "            print(\"**          Check convert-hf-to-gguf-update.py and update it accordingly.\")\n"
src_func +=  "            print(\"**\")\n"
src_func +=  "            print(f\"** chkhsh:  {chkhsh}\")\n"
src_func +=  "            print(\"**************************************************************************************\")\n"
src_func +=  "            print(\"\\n\")\n"
src_func +=  "            raise NotImplementedError(\"BPE pre-tokenizer was not recognized - update get_vocab_base_pre()\")\n"
src_func +=  "\n"
src_func +=  "        print(f\"tokenizer.ggml.pre: {res}\")\n"
src_func +=  "        print(f\"chkhsh: {chkhsh}\")\n"
src_func +=  "\n"
src_func +=  "        return res\n"

print(src_func)

print("\n")
print("!!! Copy-paste the function above into convert-hf-to-gguf.py !!!")
print("\n")

# generate tests for each tokenizer model

tests = [
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\n\n",
    "\n\n\n",
    "\t\n",
    "Hello world",
    " Hello world",
    "Hello World",
    " Hello World",
    " Hello World!",
    "Hello, world!",
    " Hello, world!",
    " this is 🦙.cpp",
    "w048 7tuijk dsdfhu",
    "нещо на Български",
    "កាន់តែពិសេសអាចខលចេញ",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
    "Hello",
    " Hello",
    "  Hello",
    "   Hello",
    "    Hello",
    "    Hello\n    Hello",
    " (",
    "\n =",
    "' era",
    "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
    "3",
    "33",
    "333",
    "3333",
    "33333",
    "333333",
    "3333333",
    "33333333",
    "333333333",
    chktxt,
]

# write the tests to ./models/ggml-vocab-{name}.gguf.inp
# the format is:
#
# test0
# __ggml_vocab_test__
# test1
# __ggml_vocab_test__
# ...
#

# with each model, encode all tests and write the results in ./models/ggml-vocab-{name}.gguf.out
# for each test, write the resulting tokens on a separate line

for model in models:
    name = model["name"]
    tokt = model["tokt"]

    # create the tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"models/tokenizers/{name}")

    with open(f"models/ggml-vocab-{name}.gguf.inp", "w") as f:
        for text in tests:
            f.write(f"{text}")
            f.write("\n__ggml_vocab_test__\n")

    with open(f"models/ggml-vocab-{name}.gguf.out", "w") as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f" {r}")
            f.write("\n")

    print(f"Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*")

# generate commands for creating vocab files

print("\nRun the following commands to generate the vocab files for testing:\n")

for model in models:
    name = model["name"]

    print(f"python3 convert-hf-to-gguf.py models/tokenizers/{name}/ --outfile models/ggml-vocab-{name}.gguf --vocab-only")

print("\n")
