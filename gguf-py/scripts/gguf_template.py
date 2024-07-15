#!/usr/bin/env python3
"""
gguf_template.py - example file to extract the chat template from the models metadata
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import jinja2
import jinja2.sandbox

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import Keys
from gguf.gguf_reader import GGUFReader  # noqa: E402

# Configure logging
logger = logging.getLogger("gguf-chat-template")


def get_chat_template(model_file: str) -> str:
    reader = GGUFReader(model_file)

    # Available keys
    logger.debug("Detected model metadata!")
    logger.debug("Outputting available model fields:")
    for key in reader.fields.keys():
        logger.debug(key)

    # Access the 'chat_template' field directly using its key
    chat_template_field = reader.fields.get(Keys.Tokenizer.CHAT_TEMPLATE)

    if chat_template_field:
        # Extract the chat template string from the field
        chat_template_memmap = chat_template_field.parts[-1]
        chat_template_string = chat_template_memmap.tobytes().decode("utf-8")
        return chat_template_string
    else:
        logger.error("Chat template field not found in model metadata.")
        return ""


def render_chat_template(
    chat_template: str,
    bos_token: str,
    eos_token: str,
    add_generation_prompt: bool = False,
    render_template: bool = False,
) -> str:
    """
    Display the chat template to standard output, optionally formatting it using Jinja2.

    Args:
        chat_template (str): The extracted chat template.
        render_template (bool, optional): Whether to format the template using Jinja2. Defaults to False.
    """
    logger.debug(f"Render Template: {render_template}")
    logger.debug(f"Add Generation Prompt: {add_generation_prompt}")

    if render_template:
        # Render the formatted template using Jinja2 with a context that includes 'bos_token' and 'eos_token'
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True
        )
        template = env.from_string(chat_template)

        messages = [
            {"role": "system", "content": "I am a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How may I assist you today?"},
            {"role": "user", "content": "Can you tell me what pickled mayonnaise is?"},
            {"role": "assistant", "content": "Certainly! What would you like to know about it?"},
            {"role": "user", "content": "Is it just regular mayonnaise with vinegar or something else?"},
        ]

        try:
            formatted_template = template.render(
                messages=messages,
                bos_token=bos_token,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt,
            )
        except jinja2.exceptions.UndefinedError:
            # system message is incompatible with set format
            formatted_template = template.render(
                messages=messages[1:],
                bos_token=bos_token,
                eos_token=eos_token,
                add_generation_prompt=add_generation_prompt,
            )

        return formatted_template
    else:
        # Display the raw template
        return chat_template


# Example usage:
def main():
    parser = argparse.ArgumentParser(
        description="Extract chat template from a GGUF model file"
    )
    parser.add_argument("model_file", type=str, help="Path to the GGUF model file")
    parser.add_argument(
        "-r",
        "--render-template",
        action="store_true",
        help="Render the chat template using Jinja2. Default is False.",
    )
    parser.add_argument(
        "-b",
        "--bos",
        default="<s>",
        help="Set a bos special token. Default is '<s>'.",
    )
    parser.add_argument(
        "-e",
        "--eos",
        default="</s>",
        help="Set a eos special token. Default is '</s>'.",
    )
    parser.add_argument(
        "-g",
        "--agp",
        action="store_true",
        help="Add generation prompt. Default is False.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output model keys. Default is False.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    chat_template = get_chat_template(args.model_file)
    rendered_template = render_chat_template(
        chat_template,
        args.bos,
        args.eos,
        add_generation_prompt=args.agp,
        render_template=args.render_template,
    )
    print(rendered_template)  # noqa: NP100


if __name__ == "__main__":
    main()
