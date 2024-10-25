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
    python ./scripts/update_jinja_goldens.py

  https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py
'''

import logging
import datetime
import glob
import os
from huggingface_hub import hf_hub_download
import json
import jinja2
import jinja2.ext
import re
# import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

model_ids = [
    "abacusai/Fewshot-Metamath-OrcaVicuna-Mistral",
    "bofenghuang/vigogne-2-70b-chat",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    "deepseek-ai/DeepSeek-V2.5",
    "indischepartij/MiniCPM-3B-OpenHermes-2.5-v2",
    "meetkai/functionary-medium-v3.1",
    "meetkai/functionary-medium-v3.2",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3.5-vision-instruct",
    "mlabonne/AlphaMonarch-7B",
    "CohereForAI/c4ai-command-r-plus",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "NousResearch/Hermes-2-Pro-Mistral-7B",
    "NousResearch/Hermes-3-Llama-3.1-70B",
    "openchat/openchat-3.5-0106",
    "OrionStarAI/Orion-14B-Chat",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "TheBloke/FusionNet_34Bx2_MoE-AWQ",

    # Gated models:
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-7b-it",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


def raise_exception(message: str):
    raise ValueError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


TEST_DATE = os.environ.get('TEST_DATE', '2024-07-26')


def strftime_now(format):
    now = datetime.datetime.strptime(TEST_DATE, "%Y-%m-%d")
    # now = datetime.datetime.now()
    return now.strftime(format)


def handle_chat_template(model_id, variant, template_src):
    logger.info(f"# {model_id}{' @ ' + variant if variant else ''}")
    model_name = model_id.replace("/", "-")
    base_name = f'{model_name}-{variant}' if variant else model_name
    template_file = f'tests/chat/templates/{base_name}.jinja'
    logger.info(f'- template_file: {template_file}')
    with open(template_file, 'w') as f:
        f.write(template_src)

    logger.info(f"- {template_file}")

    env = jinja2.Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        # keep_trailing_newline=False,
        extensions=[
            jinja2.ext.loopcontrols
        ])
    env.filters['safe'] = lambda x: x
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
        logger.info(f"- {output_file}")

        # The template (and workarounds) may modify the context in place, so we need to make a copy of it.
        render_context = json.loads(json.dumps(context))

        # Work around Llama-3.1 template quirk: it expects tool_call.function.arguments to be an object rather than its JSON string representation.
        if 'tool_call.arguments | items' in template_src or 'tool_call.arguments | tojson' in template_src:
            for message in render_context['messages']:
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        if tool_call.get('type') == 'function':
                            arguments = tool_call['function']['arguments']
                            tool_call['function']['arguments'] = json.loads(arguments)

        try:
            output = template.render(**render_context)
        except Exception as e1:
            # Some templates (e.g. Phi-3-medium-128k's) expect a non-null "content" key in each message.
            for message in context["messages"]:
                if message.get("content") is None:
                    message["content"] = ""

            try:
                output = template.render(**render_context)
            except Exception as e2:
                logger.info(f"  ERROR: {e2} (after first error: {e1})")
                output = f"ERROR: {e2}"

        with open(output_file, 'w') as f:
            f.write(output)

    logger.info('')


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
