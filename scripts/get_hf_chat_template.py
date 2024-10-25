'''
  Fetches the Jinja chat template of a HuggingFace model.
  If a model

  Syntax:
    get_hf_chat_template.py model_id [variant]

  Examples:
    python ./scripts/get_hf_chat_template.py NousResearch/Meta-Llama-3-8B-Instruct
    python ./scripts/get_hf_chat_template.py NousResearch/Hermes-3-Llama-3.1-70B tool_use
    python ./scripts/get_hf_chat_template.py meta-llama/Llama-3.2-3B-Instruct
'''

import json
import re
import sys


def main(args):
    if len(args) < 1:
        raise ValueError("Please provide a model ID and an optional variant name")
    model_id = args[0]
    variant = None if len(args) < 2 else args[1]

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
        print(chat_template, end=None)
    else:
        variants = {
            ct['name']: ct['template']
            for ct in chat_template
        }
        format_variants = lambda: ', '.join(f'"{v}"' for v in variants.keys())

        if variant is None:
            if 'default' not in variants:
                raise Exception(f'Please specify a chat template variant (one of {format_variants()})')
            variant = 'default'
            print(f'Note: picked "default" chat template variant (out of {format_variants()})', file=sys.stderr)
        elif variant not in variants:
            raise Exception(f"Variant {variant} not found in chat template (found {format_variants()})")

        print(variants[variant], end=None)


if __name__ == '__main__':
    main(sys.argv[1:])
