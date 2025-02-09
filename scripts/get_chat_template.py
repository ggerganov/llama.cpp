#!/usr/bin/env python
'''
  Fetches the Jinja chat template of a HuggingFace model.
  If a model has multiple chat templates, you can specify the variant name.

  Syntax:
    ./scripts/get_chat_template.py model_id [variant]

  Examples:
    ./scripts/get_chat_template.py CohereForAI/c4ai-command-r-plus tool_use      | tee models/templates/CohereForAI-c4ai-command-r-plus-tool_use.jinja
    ./scripts/get_chat_template.py CohereForAI/c4ai-command-r7b-12-2024 default  | tee models/templates/CohereForAI-c4ai-command-r7b-12-2024-default.jinja
    ./scripts/get_chat_template.py CohereForAI/c4ai-command-r7b-12-2024 rag      | tee models/templates/CohereForAI-c4ai-command-r7b-12-2024-rag.jinja
    ./scripts/get_chat_template.py CohereForAI/c4ai-command-r7b-12-2024 tool_use | tee models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja
    ./scripts/get_chat_template.py deepseek-ai/DeepSeek-R1-Distill-Llama-8B      | tee models/templates/deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jinja
    ./scripts/get_chat_template.py deepseek-ai/DeepSeek-R1-Distill-Qwen-32B      | tee models/templates/deepseek-ai-DeepSeek-R1-Distill-Qwen-32B.jinja
    ./scripts/get_chat_template.py fireworks-ai/llama-3-firefunction-v2          | tee models/templates/fireworks-ai-llama-3-firefunction-v2.jinja
    ./scripts/get_chat_template.py google/gemma-2-2b-it                          | tee models/templates/google-gemma-2-2b-it.jinja
    ./scripts/get_chat_template.py meetkai/functionary-medium-v3.                | tee models/templates/meetkai-functionary-medium-v3.jinja
    ./scripts/get_chat_template.py meetkai/functionary-medium-v3.2               | tee models/templates/meetkai-functionary-medium-v3.2.jinja
    ./scripts/get_chat_template.py meta-llama/Llama-3.1-8B-Instruct              | tee models/templates/meta-llama-Llama-3.1-8B-Instruct.jinja
    ./scripts/get_chat_template.py meta-llama/Llama-3.2-3B-Instruct              | tee models/templates/meta-llama-Llama-3.2-3B-Instruct.jinja
    ./scripts/get_chat_template.py meta-llama/Llama-3.3-70B-Instruct             | tee models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja
    ./scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct               | tee models/templates/microsoft-Phi-3.5-mini-instruct.jinja
    ./scripts/get_chat_template.py mistralai/Mistral-Nemo-Instruct-2407          | tee models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja
    ./scripts/get_chat_template.py NousResearch/Hermes-2-Pro-Llama-3-8B tool_use | tee models/templates/NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja
    ./scripts/get_chat_template.py NousResearch/Hermes-3-Llama-3.1-8B tool_use   | tee models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja
    ./scripts/get_chat_template.py Qwen/Qwen2.5-7B-Instruct                      | tee models/templates/Qwen-Qwen2.5-7B-Instruct.jinja
'''

import json
import re
import sys


def get_chat_template(model_id, variant=None):
    try:
        # Use huggingface_hub library if available.
        # Allows access to gated models if the user has access and ran `huggingface-cli login`.
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")) as f:
            config_str = f.read()
    except ImportError:
        import requests
        assert re.match(r"^[\w.-]+/[\w.-]+$", model_id), f"Invalid model ID: {model_id}"
        response = requests.get(f"https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json")
        if response.status_code == 401:
            raise Exception('Access to this model is gated, please request access, authenticate with `huggingface-cli login` and make sure to run `pip install huggingface_hub`')
        response.raise_for_status()
        config_str = response.text

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        # Fix https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
        # (Remove extra '}' near the end of the file)
        config = json.loads(re.sub(r'\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")', r'\1', config_str))

    chat_template = config['chat_template']
    if isinstance(chat_template, str):
        return chat_template
    else:
        variants = {
            ct['name']: ct['template']
            for ct in chat_template
        }

        def format_variants():
            return ', '.join(f'"{v}"' for v in variants.keys())

        if variant is None:
            if 'default' not in variants:
                raise Exception(f'Please specify a chat template variant (one of {format_variants()})')
            variant = 'default'
            sys.stderr.write(f'Note: picked "default" chat template variant (out of {format_variants()})\n')
        elif variant not in variants:
            raise Exception(f"Variant {variant} not found in chat template (found {format_variants()})")

        return variants[variant]


def main(args):
    if len(args) < 1:
        raise ValueError("Please provide a model ID and an optional variant name")
    model_id = args[0]
    variant = None if len(args) < 2 else args[1]

    template = get_chat_template(model_id, variant)
    sys.stdout.write(template)


if __name__ == '__main__':
    main(sys.argv[1:])
