#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.huggingface_hub import HFVocabRequest

logger = logging.getLogger("gguf-gen-pre")


def test_pre_tok(content) -> None:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="A huggingface read auth token"
    )
    parser.add_argument(
        "-m", "--model-path", default=None, help="The models storage path"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hf_vocab_req = HFVocabRequest(
        args.model_path, args.hf_auth_token, logger
    )

if __name__ == '__main__':
    main()
