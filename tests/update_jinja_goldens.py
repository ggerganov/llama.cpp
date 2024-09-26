#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jinja2",
#     "huggingface_hub",
# ]
# ///
'''
  Fetches the Jinja2 templates of a few known models and use them to generate prompt goldens for a few predefined chat contexts.

  Examples:
    python ./tests/update_jinja_goldens.py

  https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py
'''

import datetime
import glob
import os
from huggingface_hub import hf_hub_download
import json
import jinja2
import jinja2.ext
import re
# import requests

model_ids = [
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "meetkai/functionary-medium-v3.2",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "indischepartij/MiniCPM-3B-OpenHermes-2.5-v2",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "TheBloke/FusionNet_34Bx2_MoE-AWQ",
    "bofenghuang/vigogne-2-70b-chat",
    "mlabonne/AlphaMonarch-7B",
    "OrionStarAI/Orion-14B-Chat",
    "openchat/openchat-3.5-0106",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "abacusai/Fewshot-Metamath-OrcaVicuna-Mistral",
    "CohereForAI/c4ai-command-r-plus",
    "THUDM/chatglm3-6b",
    "derek33125/project-angel-chatglm4",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/DeepSeek-V2.5",

    # Needs debugging:
    # "eachadea/vicuna-13b-1.1",
    # "microsoft/Phi-3-vision-instruct",

    # Gated models:
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-7b-it",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


def raise_exception(message: str):
    raise ValueError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def strftime_now(format):
    return datetime.now().strftime(format)


def handle_chat_template(model_id, variant, template_src):
    print(f"# {model_id} @ {variant}", flush=True)
    model_name = model_id.replace("/", "-")
    base_name = f'{model_name}-{variant}' if variant else model_name
    template_file = f'tests/chat/templates/{base_name}.jinja'
    print(f'template_file: {template_file}')
    with open(template_file, 'w') as f:
        f.write(template_src)

    print(f"- {template_file}", flush=True)

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        # keep_trailing_newline=False,
        extensions=[
            jinja2.ext.loopcontrols
        ])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    env.globals['strftime_now'] = strftime_now

    template_handles_tools = 'tools' in template_src
    template_hates_the_system = 'System role not supported' in template_src

    template = env.from_string(template_src)

    context_files = glob.glob('tests/chat/contexts/*.json')
    for context_file in context_files:
        context_name = context_file.split("/")[-1].replace(".json", "")
        with open(context_file, 'r') as f:
            context = json.load(f)

        if not template_handles_tools and 'tools' in context:
            continue

        if template_hates_the_system and any(m['role'] == 'system' for m in context['messages']):
            continue

        output_file = f'tests/chat/goldens/{base_name}-{context_name}.txt'
        print(f"- {output_file}", flush=True)
        try:
            output = template.render(**context)
        except Exception as e1:
            # Some templates (e.g. Phi-3-medium-128k's) expect a non-null "content" key in each message.
            for message in context["messages"]:
                if message.get("content") is None:
                    message["content"] = ""

            try:
                output = template.render(**context)
            except Exception as e2:
                print(f"  ERROR: {e2} (after first error: {e1})", flush=True)
                output = f"ERROR: {e2}"

        with open(output_file, 'w') as f:
            f.write(output)

    print()


def main():
    for dir in ['tests/chat/templates', 'tests/chat/goldens']:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    for model_id in model_ids:
        # response = requests.get(f"https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json")
        # response.raise_for_status()
        # config_str = response.text
        with open(hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")) as f:
            config_str = f.read()

        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            # Fix https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
            # (Remove extra '}' near the end of the file)
            config = json.loads(re.sub(r'\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")', r'\1', config_str))

        chat_template = config['chat_template']
        if isinstance(chat_template, str):
            handle_chat_template(model_id, None, chat_template)
        else:
            for ct in chat_template:
                handle_chat_template(model_id, ct['name'], ct['template'])


if __name__ == '__main__':
    main()
