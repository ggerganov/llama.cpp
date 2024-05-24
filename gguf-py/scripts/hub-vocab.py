#!/usr/bin/env python3

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
from gguf.huggingface_hub import HFVocabRequest

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
        default="models/",
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

    vocab_request = HFVocabRequest(args.auth_token, args.model_path, logger)
    vocab_type = vocab_request.get_vocab_enum(args.vocab_type)
    vocab_request.get_all_vocab_files(args.model_repo, vocab_type)


if __name__ == "__main__":
    main()
