#!/usr/bin/env python3
"""
Tokenizers Vocabulary Notes:

Normalizers:
Normalizers are a set of operations applied to raw string input data to make it less random or “cleaner”. Common normalization operations include stripping whitespace, removing accented characters or lowercasing all text. The Hugging Face `tokenizers` library provides various Normalizer classes that can be combined using a normalizers.Sequence to apply multiple normalization operations in sequence on the input data before tokenization takes place.

Pre-Tokenization:
Pre-Tokenization encompasses identifying characters and their types, including letters, numbers, whitespace, etc., prior to applying actual tokenization or feeding the data into machine learning models. The Hugging Face `tokenizers` library provides several Pre-tokenizer classes that can be used for different purposes such as Byte Level pre-tokenization (using openai/gpt-2 RegEx by default) and BERT pre-tokenization, which inherits from Byte Level tokenization but has some differences in its behavior.

Pre-Tokenization Types:

1. Byte Level Pre-tokenization:
   - Default regular expression used for pattern matching is taken from openai/gpt-2 `encoder.py`.

2. BERT pre-tokenization (inherits from Byte Level):
   - Differences in behavior compared to the default Byte Level tokenizer, but defaults for each RegEx are identical in either case.

Pre-Tokenization Character Types:

1. Sequence: Matches a sequence of characters that should be treated as a single unit during preprocessing or tokenization.
2. Letters and Numbers (Alphabetic/Alphanumeric): Characters belonging to the alphabet or mixed combinations of letters and numbers, respectively.
3. Whitespace: Spaces, tabs, newlines, etc., that separate words or other units in the text data.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if (
    "NO_LOCAL_GGUF" not in os.environ
    and (Path(__file__).parent.parent.parent / "gguf-py").exists()
):
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.huggingface_hub import HFHubModel, HFHubTokenizer

logger = logging.getLogger(Path(__file__).stem)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "model_repo", help="A huggingface model repository, e.g. org/model"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "--model-path",
        default="models",
        help="The models storage path. Default is 'models/'.",
    )
    parser.add_argument(
        "--vocab-name",
        const="BPE",
        nargs="?",
        choices=["SPM", "BPE", "WPM"],
        help="The name of the vocab type. Default is 'BPE'.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hub_model = HFHubModel(
        auth_token=args.auth_token,
        model_path=args.model_path,
        logger=logger,
    )

    hub_tokenizer = HFHubTokenizer(
        model_path=args.model_path,
        logger=logger,
    )

    vocab_type = HFHubTokenizer.get_vocab_type(args.vocab_name)
    hub_model.download_all_vocab_files(
        model_repo=args.model_repo,
        vocab_type=vocab_type,
    )

    hub_model.download_all_vocab_files(args.model_repo, vocab_type)
    hub_tokenizer.log_tokenizer_json_info(args.model_repo)


if __name__ == "__main__":
    main()
