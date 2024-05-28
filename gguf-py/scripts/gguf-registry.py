#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import (
    HF_TOKENIZER_BPE_FILES,
    HF_TOKENIZER_SPM_FILES,
    MODEL_ARCH,
    MODEL_ARCH_NAMES,
    ModelFileExtension,
    PreTokenizerType,
    VocabType,
)
from gguf.huggingface_hub import HFHubModel, HFHubTokenizer

logger = logging.getLogger(__file__)

#
# HuggingFace Model Map
#
# NOTE: All prerequisite model metadata must be defined here.
#
# Defines metadata for each Hugging Face model required during conversion to GGUF
#
# Field Descriptions
#   - `model_repo` (str): The HuggingFace endpoint or local path to the models repository
#   - `model_arch` (MODEL_ARCH): Model architecture type
#   - `model_parts` (int): Number of parts required to join the model during conversion
#   - `model_type` (FileFormatType): File format for the Hugging Face model files
#   - `vocab_type` (VocabType): Vocabulary type used by the tokenizer
#   - `vocab_pre` (Optional[Tuple[str]]): Tuple of pre-tokenizer pattern strings for this model
#   - `vocab_files` (Tuple[str]): Tuple of file names required to extract vocabulary and other metadata
#
# NOTES
#   - Possible algorithms are WordLevel, BPE, WordPiece, or Unigram
#   - Possible LLaMa tokenizer model types are: None, SPM, BPE, or WPM
HF_MODEL_MAP = (
    # SPM (Sentence Piece Models): Default to Byte Level Pre-tokenization.
    {
        "model_repo": "meta-llama/Llama-2-7b-hf",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.1",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 3,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {  # NOTE: Mistral v0.3 has a 'tokenizer.model.v3' file
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 3,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 8,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "microsoft/Phi-3-mini-4k-instruct",
        "model_arch": MODEL_ARCH.PHI3,
        "model_parts": 2,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.SPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    # WPM (Word Piece Models): Default to Byte Level Pre-tokenization.
    # NOTE: BERT Normalization and Pre-tokenization rules differ from Byte Level Pre-tokenization.
    {
        "model_repo": "BAAI/bge-small-en-v1.5",
        "model_arch": MODEL_ARCH.BERT,
        "model_parts": 1,
        "model_type": ModelFileExtension.BIN.value,
        "vocab_type": VocabType.WPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "jinaai/jina-embeddings-v2-base-en",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.WPM.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    # BPE (Byte Pair Encoding Models): Default is Byte Level Pre-tokenization
    {
        "model_repo": "meta-llama/Meta-Llama-3-8B",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 4,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "tiiuae/falcon-7b",
        "model_arch": MODEL_ARCH.FALCON,
        "model_parts": 2,
        "model_type": ModelFileExtension.BIN.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "deepseek-ai/deepseek-llm-7b-base",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileExtension.BIN.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "deepseek-ai/deepseek-coder-6.7b-base",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "mosaicml/mpt-7b",
        "model_arch": MODEL_ARCH.MPT,
        "model_parts": 2,
        "model_type": ModelFileExtension.BIN.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    #
    # BPE: STARCODER
    #
    {
        "model_repo": "bigcode/starcoder2-3b",
        "model_arch": MODEL_ARCH.STARCODER2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "smallcloudai/Refact-1_6-base",
        "model_arch": MODEL_ARCH.REFACT,
        "model_parts": 1,
        "model_type": ModelFileExtension.BIN.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "CohereForAI/c4ai-command-r-v01",
        "model_arch": MODEL_ARCH.COMMAND_R,
        "model_parts": 15,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    #
    # BPE: QWEN
    #
    {
        "model_repo": "Qwen/Qwen1.5-7B",
        "model_arch": MODEL_ARCH.QWEN2,
        "model_parts": 4,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "stabilityai/stablelm-2-zephyr-1_6b",
        "model_arch": MODEL_ARCH.STABLELM,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    #
    # BPE: GPT-2
    #
    {
        "model_repo": "openai-community/gpt2",
        "model_arch": MODEL_ARCH.GPT2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "allenai/OLMo-1.7-7B-hf",
        "model_arch": MODEL_ARCH.OLMO,
        "model_parts": 6,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    # {  # NOTE: I don't have access to this model
    #     "model_repo": "databricks/dbrx-base",
    #     "model_arch": MODEL_ARCH.DBRX,
    #     "model_parts": 0,
    #     "model_type": ModelFileExtension.SAFETENSORS.value,
    #     "vocab_type": VocabType.BPE.value,
    #     "vocab_pre": None,
    #     "vocab_files": HF_TOKENIZER_BPE_FILES,
    # },
    {  # NOTE: RoBERTa post processor
        "model_repo": "jinaai/jina-embeddings-v2-base-es",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: RoBERTa post processor
        "model_repo": "jinaai/jina-embeddings-v2-base-de",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: Phi-1 is compatible with GPT-2 arch and vocab
        "model_repo": "microsoft/phi-1",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "microsoft/phi-1_5",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 1,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "microsoft/phi-2",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 2,
        "model_type": ModelFileExtension.SAFETENSORS.value,
        "vocab_type": VocabType.BPE.value,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "--model-path",
        default="models",
        help="The models storage path. Default is 'models'.",
    )
    return parser.parse_args()


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


metadata = []
for model in HF_MODEL_MAP:
    model_repo = model["model_repo"]
    model_arch = model["model_arch"]
    vocab_type = model["vocab_type"]

    print("HUB_REPO:", model_repo, "LLAMA_ARCH:", MODEL_ARCH_NAMES[model_arch])

    hub_model.download_all_vocab_files(
        model_repo=model_repo,
        vocab_type=vocab_type,
    )
    # log the downloaded results
    hub_tokenizer.log_tokenizer_json_info(model_repo)

    model["model_arch"] = MODEL_ARCH_NAMES[model_arch]

    normalizer = hub_tokenizer.get_normalizer(model_repo)
    # NOTE: Normalizer may be one of null, Sequence, NFC, NFD, NFKC, NFKD...
    # Seems to be null, Sequence, or NFC in most cases
    # Default to NFD
    # TODO: Extract the normalizer metadata
    model["normalizer"] = normalizer

    # Seems safe to assume most basic types are of type "Sequence"
    # I expect this to cause issues in the future. Needs more research.
    pre_tokenizer = hub_tokenizer.get_pre_tokenizer(model_repo)
    # extract the added tokens metadata
    model["pre_tokenizer"] = pre_tokenizer

    added_tokens = hub_tokenizer.get_added_tokens(model_repo)
    # extract the added tokens metadata
    model["added_tokens"] = added_tokens

    sha256sum = hub_tokenizer.get_tokenizer_json_hash(model_repo)
    # use the hash to validate the models vocabulary
    model["vocab_hash"] = sha256sum

    metadata.append(model)

with open(f"{args.model_path}/registry.json", mode="w") as file:
    json.dump(metadata, file, indent=2)
