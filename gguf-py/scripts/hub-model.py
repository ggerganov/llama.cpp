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
        "-m", "--model-path", default="models",
        help="The models storage path. Default is 'models'.",
    )
    parser.add_argument(
        "-a", "--model-arch", default="llama",
        help="The supported llama.cpp model architecture. Default is 'llama'."
    )
    parser.add_argument(
        "-p", "--model-parts", default=2,
        help="The number of model shards encompassing the model. Default is 2."
    )
    parser.add_argument(
        "-f", "--model-name",
        default=".safetensors", const=".safetensors", nargs="?",
        choices=[".pt", ".pth", ".bin", ".safetensors", ".gguf"],
        help="The models file name extension. Default is '.safetensors'"
    )
    parser.add_argument(
        "-t", "--vocab-type",
        nargs="?", choices=["SPM", "BPE", "WPM"],
        help="The models tokenizer type. Default is 'SPM'."
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

    model_type = HFHubModel.get_model_type(args.model_name)
    hub_model.download_model_files(args.model_repo, model_type)


if __name__ == '__main__':
    main()
