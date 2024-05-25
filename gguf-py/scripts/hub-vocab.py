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

from gguf.huggingface_hub import HFHubModel, HFHubTokenizer

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
    args = parser.parse_args()

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
