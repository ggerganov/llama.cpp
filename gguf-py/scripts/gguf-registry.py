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

from gguf.constants import (
    GPT_PRE_TOKENIZER_DEFAULT,
    HF_TOKENIZER_BPE_FILES,
    HF_TOKENIZER_SPM_FILES,
    MODEL_ARCH,
    MODEL_ARCH_NAMES,
    ModelFileType,
    VocabType,
)
from gguf.huggingface_hub import HFHubModel, HFHubTokenizer

logger = logging.getLogger("gguf-gen-pre")

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
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.SPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 3,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.SPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 8,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.SPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    {
        "model_repo": "microsoft/Phi-3-mini-4k-instruct",
        "model_arch": MODEL_ARCH.PHI3,
        "model_parts": 2,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.SPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_SPM_FILES,
    },
    # WPM (Word Piece Models): Default to Byte Level Pre-tokenization.
    # NOTE: BERT Normalization and Pre-tokenization rules differ from Byte Level Pre-tokenization.
    {
        "model_repo": "BAAI/bge-small-en-v1.5",
        "model_arch": MODEL_ARCH.BERT,
        "model_parts": 1,
        "model_type": ModelFileType.BIN,
        "vocab_type": VocabType.WPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "jinaai/jina-embeddings-v2-base-en",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.WPM,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    # BPE (Byte Pair Encoding Models): Default is Byte Level Pre-tokenization
    {
        "model_repo": "meta-llama/Meta-Llama-3-8B",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 4,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "tiiuae/falcon-7b",
        "model_arch": MODEL_ARCH.FALCON,
        "model_parts": 2,
        "model_type": ModelFileType.BIN,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "deepseek-ai/deepseek-llm-7b-base",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileType.BIN,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "deepseek-ai/deepseek-coder-6.7b-base",
        "model_arch": MODEL_ARCH.LLAMA,
        "model_parts": 2,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "mosaicml/mpt-7b",
        "model_arch": MODEL_ARCH.MPT,
        "model_parts": 2,
        "model_type": ModelFileType.BIN,
        "vocab_type": VocabType.BPE,
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
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "smallcloudai/Refact-1_6-base",
        "model_arch": MODEL_ARCH.REFACT,
        "model_parts": 1,
        "model_type": ModelFileType.BIN,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "CohereForAI/c4ai-command-r-v01",
        "model_arch": MODEL_ARCH.COMMAND_R,
        "model_parts": 15,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
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
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "stabilityai/stablelm-2-zephyr-1_6b",
        "model_arch": MODEL_ARCH.STABLELM,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
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
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "allenai/OLMo-1.7-7B-hf",
        "model_arch": MODEL_ARCH.OLMO,
        "model_parts": 6,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: I don't have access to this model
        "model_repo": "databricks/dbrx-base",
        "model_arch": MODEL_ARCH.DBRX,
        "model_parts": 0,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: RoBERTa post processor
        "model_repo": "jinaai/jina-embeddings-v2-base-es",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: RoBERTa post processor
        "model_repo": "jinaai/jina-embeddings-v2-base-de",
        "model_arch": MODEL_ARCH.JINA_BERT_V2,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {  # NOTE: Phi-1 is compatible with GPT-2 arch and vocab
        "model_repo": "microsoft/phi-1",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "microsoft/phi-1_5",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 1,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
    {
        "model_repo": "microsoft/phi-2",
        "model_arch": MODEL_ARCH.PHI2,
        "model_parts": 2,
        "model_type": ModelFileType.SAFETENSORS,
        "vocab_type": VocabType.BPE,
        "vocab_pre": None,
        "vocab_files": HF_TOKENIZER_BPE_FILES,
    },
)
