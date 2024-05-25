#!/usr/bin/env python3
"""
Tokenizers Vocabulary Notes:

Normalizers:
TODO

Pre-tokenizers:

Byte Level Pre-tokenization uses openai/gpt-2 RegEx from `encoder.py` by default.
There are other Pre-tokenization types, e.g. BERT, which inherits from Byte Level
The defaults for each RegEx are identical in either case.

Pre-Tokenization encompasses identify characters and their types
- A pattern may match a type of "Sequence"
- Letters and Numbers: Alphabetic or Alphanumeric
- Whitespace:
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

from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES
from gguf.huggingface_hub import HFHub, HFTokenizer

logger = logging.getLogger(Path(__file__).stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "model_repo", help="A huggingface model repository, e.g. org/model"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        default="models",
        help="The models storage path. Default is 'models/'.",
    )
    parser.add_argument(
        "--vocab-type",
        const="BPE",
        nargs="?",
        choices=["SPM", "BPE", "WPM"],
        help="The type of vocab. Default is 'BPE'.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    vocab_request = HFModel(args.auth_token, args.model_path, logger)
    vocab_type = HFTokenizer.get_vocab_enum(args.vocab_type)
    tokenizer = vocab_request.tokenizer
    vocab_request.get_all_vocab_files(args.model_repo, vocab_type)
    tokenizer.log_tokenizer_json_info(args.model_repo)


if __name__ == "__main__":
    main()
