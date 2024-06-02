from __future__ import annotations

import json
import frontmatter
from pathlib import Path

from typing import Optional
from dataclasses import dataclass

from .constants import Keys


@dataclass
class Metadata:
    # Authorship Metadata to be written to GGUF KV Store
    name: Optional[str] = None
    basename: Optional[str] = None
    finetune: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_link: Optional[str] = None
    source_url: Optional[str] = None
    source_hf_repo: Optional[str] = None

    @staticmethod
    def load(metadata_override_path: Path, model_path: Path) -> Metadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new Metadata instance
        metadata = Metadata()

        # load model folder model card if available
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        model_card = Metadata.load_model_card(model_path)
        if metadata.name is None:
            if "model-index" in model_card and len(model_card["model_name"]) == 1 and "name" in model_card["model_name"][0]:
                metadata.name = model_card["model_name"][0].get("name")
            elif "model_name" in model_card:
                # non huggingface model card standard but notice some model creator using it
                metadata.name = model_card.get("model_name")
        if metadata.license is None:
            metadata.license = model_card.get("license")
        if metadata.license_name is None:
            metadata.license_name = model_card.get("license_name")
        if metadata.license_link is None:
            metadata.license_link = model_card.get("license_link")

        # load huggingface parameters if available
        hf_params = Metadata.load_huggingface_parameters(model_path)
        hf_name_or_path = hf_params.get("_name_or_path")
        if metadata.name is None and hf_name_or_path is not None:
            metadata.name = Path(hf_name_or_path).name
        if metadata.source_hf_repo is None and hf_name_or_path is not None:
            metadata.source_hf_repo = Path(hf_name_or_path).name

        # Use Directory Folder Name As Fallback Name
        if metadata.name is None:
            if model_path is not None and model_path.exists():
                metadata.name = model_path.name

        # Metadata Override
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata_override = Metadata.load_metadata_override(metadata_override_path)
        metadata.name           = metadata_override.get(Keys.General.NAME          ,  metadata.name          ) # noqa: E202
        metadata.basename       = metadata_override.get(Keys.General.BASENAME      ,  metadata.basename      ) # noqa: E202
        metadata.finetune       = metadata_override.get(Keys.General.FINETUNE      ,  metadata.finetune      ) # noqa: E202
        metadata.author         = metadata_override.get(Keys.General.AUTHOR        ,  metadata.author        ) # noqa: E202
        metadata.version        = metadata_override.get(Keys.General.VERSION       ,  metadata.version       ) # noqa: E202
        metadata.url            = metadata_override.get(Keys.General.URL           ,  metadata.url           ) # noqa: E202
        metadata.description    = metadata_override.get(Keys.General.DESCRIPTION   ,  metadata.description   ) # noqa: E202
        metadata.license        = metadata_override.get(Keys.General.LICENSE       ,  metadata.license       ) # noqa: E202
        metadata.license_name   = metadata_override.get(Keys.General.LICENSE_NAME  ,  metadata.license_name  ) # noqa: E202
        metadata.license_link   = metadata_override.get(Keys.General.LICENSE_LINK  ,  metadata.license_link  ) # noqa: E202
        metadata.source_url     = metadata_override.get(Keys.General.SOURCE_URL    ,  metadata.source_url    ) # noqa: E202
        metadata.source_hf_repo = metadata_override.get(Keys.General.SOURCE_HF_REPO,  metadata.source_hf_repo) # noqa: E202

        return metadata

    @staticmethod
    def load_metadata_override(metadata_override_path: Path):
        if metadata_override_path is None or not metadata_override_path.exists():
            return {}

        with open(metadata_override_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_model_card(model_path: Path):
        if model_path is None or not model_path.exists():
            return {}

        model_card_path = model_path / "README.md"

        if not model_card_path.exists():
            return {}

        with open(model_card_path, "r", encoding="utf-8") as f:
            return frontmatter.load(f)

    @staticmethod
    def load_huggingface_parameters(model_path: Path):
        if model_path is None or not model_path.exists():
            return {}

        config_path = model_path / "config.json"

        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
