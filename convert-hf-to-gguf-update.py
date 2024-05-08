#!/usr/bin/env python3

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

import json
import logging
import os
import sys
from enum import IntEnum, auto
from hashlib import sha256

import requests
from transformers import AutoTokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("convert-hf-to-gguf-update")


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
    logger.info("Usage: python convert-hf-to-gguf-update.py <huggingface_token>")
    sys.exit(1)

# TODO: add models here, base models preferred
models = [
    {"name": "llama-spm",      "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/meta-llama/Llama-2-7b-hf", },
    {"name": "llama-bpe",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/meta-llama/Meta-Llama-3-8B", },
    {"name": "phi-3",          "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct", },
    {"name": "deepseek-llm",   "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-llm-7b-base", },
    {"name": "deepseek-coder", "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base", },
    {"name": "falcon",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/tiiuae/falcon-7b", },
    {"name": "bert-bge",       "tokt": TOKENIZER_TYPE.WPM, "repo": "https://huggingface.co/BAAI/bge-small-en-v1.5", },
    {"name": "mpt",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mosaicml/mpt-7b", },
    {"name": "starcoder",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/bigcode/starcoder2-3b", },
    {"name": "gpt-2",          "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/openai-community/gpt2", },
    {"name": "phi",            "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/microsoft/phi-1", },
    {"name": "stablelm",       "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b", },
    {"name": "qwen",           "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/Qwen/Qwen-7B", },
    {"name": "mistral-bpe",    "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2", },
    {"name": "mistral-spm",    "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2", },
    {"name": "mixtral-bpe",    "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1", },
    {"name": "mixtral-spm",    "tokt": TOKENIZER_TYPE.SPM, "repo": "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1", },
    {"name": "refact",         "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/smallcloudai/Refact-1_6-base", },
    {"name": "command-r",      "tokt": TOKENIZER_TYPE.BPE, "repo": "https://huggingface.co/CohereForAI/c4ai-command-r-v01", },
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
        logger.info(f"File {save_path} downloaded successfully")
    else:
        logger.info(f"Failed to download file. Status code: {response.status_code}")


# download the tokenizer models
for model in models:
    # set mapping
    name = model["name"]
    repo = model["repo"]
    tokt = model["tokt"]

    # set url paths
    url_main = f"{repo}/raw/main"
    url_resolve = f"{repo}/resolve/main"

    # set dir paths
    model_name_or_path = f"models/tokenizers/{name}"
    model_tokenizer_path = f"{model_name_or_path}/tokenizer.json"
    model_config_path = f"{model_name_or_path}/config.json"

    # check dir path
    if not os.path.exists(model_name_or_path):
        os.makedirs(model_name_or_path)
    else:
        logger.info(f"Directory {model_name_or_path} already exists - skipping")
        continue

    logger.info(f"Downloading {name} to {model_name_or_path}")

    # model and repo urls are not the same
    # url = "https://huggingface.co/Qwen/Qwen-tokenizer/raw/main/tokenizer.json"

    # Get the models tokenizer
    download_file_with_auth(
        url=f"{url_main}/tokenizer.json",
        token=token,
        save_path=model_tokenizer_path
    )

    # Get the models hyper params
    download_file_with_auth(
        url=f"{url_main}/config.json",
        token=token,
        save_path=model_config_path
    )

    # if downloaded file is less than 1KB, we likely need to download an LFS instead
    if os.path.getsize(model_tokenizer_path) < 1024:
        # remove the file
        os.remove(model_tokenizer_path)
        download_file_with_auth(
            url=f"{url_resolve}/tokenizer.json",
            token=token,
            save_path=model_tokenizer_path
        )

    # Handle sentencepiece tokenizer
    if tokt == TOKENIZER_TYPE.SPM:
        download_file_with_auth(
            url=f"{url_resolve}/tokenizer.model",
            token=token,
            save_path=model_tokenizer_path
        )

    # Get the tokenizer config
    download_file_with_auth(
        url=f"{url_main}/tokenizer_config.json",
        token=token,
        save_path=f"{model_name_or_path}/tokenizer_config.json"
    )

# generate the source code for the convert-hf-to-gguf.py:get_vocab_base_pre() function:
# TODO: auto-update convert-hf-to-gguf.py with the generated function

src_ifs = ""
for model in models:
    name = model["name"]
    tokt = model["tokt"]

    if tokt == TOKENIZER_TYPE.SPM:
        continue

    # create the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    chktok = tokenizer.encode(chktxt)
    chkhsh = sha256(str(chktok).encode()).hexdigest()

    logger.info(f"model: {name}")
    logger.info(f"tokt: {tokt}")
    logger.info(f"repo: {model['repo']}")
    logger.info(f"chktok: {chktok}")
    logger.info(f"chkhsh: {chkhsh}")

    # print the "pre_tokenizer" content from the tokenizer.json
    with open(model_tokenizer_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        pre_tokenizer = cfg["pre_tokenizer"]
        logger.info("pre_tokenizer: " + json.dumps(pre_tokenizer, indent=4))

    logger.info("")

    src_ifs += f"        if chkhsh == \"{chkhsh}\":\n"
    src_ifs += f"            # ref: {model['repo']}\n"
    src_ifs += f"            res = \"{name}\"\n"

src_func = f"""
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = {repr(chktxt)}

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {{chktok}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        res = None

        # NOTE: if you get an error here, you need to update the convert-hf-to-gguf-update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
{src_ifs}
        if res is None:
            logger.warning("\\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert-hf-to-gguf-update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert-hf-to-gguf-update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggerganov/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {{chkhsh}}")
            logger.warning("**************************************************************************************")
            logger.warning("\\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {{repr(res)}}")
        logger.debug(f"chkhsh: {{chkhsh}}")

        return res
"""

print(src_func) # noqa: NP100

logger.info("\n")
logger.info("!!! Copy-paste the function above into convert-hf-to-gguf.py !!!")
logger.info("\n")

# generate tests for each tokenizer model

tests = [
    "ied 4 ½ months",
    "Führer",
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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(f"models/ggml-vocab-{name}.gguf.inp", "w", encoding="utf-8") as f:
        for text in tests:
            f.write(f"{text}")
            f.write("\n__ggml_vocab_test__\n")

    with open(f"models/ggml-vocab-{name}.gguf.out", "w") as f:
        for text in tests:
            res = tokenizer.encode(text, add_special_tokens=False)
            for r in res:
                f.write(f" {r}")
            f.write("\n")

    logger.info(f"Tests for {name} written in ./models/ggml-vocab-{name}.gguf.*")

# generate commands for creating vocab files
shscript = "#!/usr/bin/env bash\n\n"

for model in models:
    name = model["name"]
    tmpline = f"python3 convert-hf-to-gguf.py {model_name_or_path}/ --outfile models/ggml-vocab-{name}.gguf --vocab-only\n"
    shscript += tmpline
    logging.info(tmpline.strip())

with open("generate-vocab.sh", "w", encoding="utf-8") as f:
    f.writelines(shscript)
    logging.info(f"Wrote {len(shscript)} bytes to generate-vocab.sh")

logging.info("Run the following command to generate the vocab files for testing:")
logging.info("Enable execution: chmod +x generate-vocab.sh")
logging.info("Execute with ./generate-vocab.sh")
