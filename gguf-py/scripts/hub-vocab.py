#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES
from gguf.huggingface_hub import HFVocabRequest

logger = logging.getLogger("gguf-gen-pre")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "-r", "--model-repo", default="meta-llama/Llama-2-7b-hf",
        help="The models repository. Default is 'meta-llama/Llama-2-7b-hf'."
    )
    parser.add_argument(
        "-m", "--model-path", default="models/",
        help="The models storage path. Default is 'models/'."
    )
    parser.add_argument(
        "--vocab-type",
        const="BPE", nargs="?", choices=["BPE", "SPM"],
        help="The type of vocab. Default is 'BPE'."
    )
    args = parser.parse_args()

    vocab_request = HFVocabRequest(args.auth_token, args.model_path, logger)
    vocab_list = vocab_request.get_vocab_filenames(args.vocab_type)
    for vocab_file in vocab_list:
        vocab_request.get_vocab_file(args.model_repo, vocab_file, args.model_path)


if __name__ == '__main__':
    main()
