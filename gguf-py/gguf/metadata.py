from __future__ import annotations

import json
from pathlib import Path

from typing import Optional
from dataclasses import dataclass

from .constants import Keys


@dataclass
class Metadata:
    name: Optional[str] = None
    basename: Optional[str] = None
    finetune: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    licence: Optional[str] = None
    source_url: Optional[str] = None
    source_hf_repo: Optional[str] = None

    @staticmethod
    def load(metadata_path: Path) -> Metadata:
        if metadata_path is None or not metadata_path.exists():
            return Metadata()

        with open(metadata_path, 'r') as file:
            data = json.load(file)

        # Create a new Metadata instance
        metadata = Metadata()

        # Assigning values to Metadata attributes if they exist in the JSON file
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata.name = data.get(Keys.General.NAME)
        metadata.basename = data.get(Keys.General.BASENAME)
        metadata.finetune = data.get(Keys.General.FINETUNE)
        metadata.author = data.get(Keys.General.AUTHOR)
        metadata.version = data.get(Keys.General.VERSION)
        metadata.url = data.get(Keys.General.URL)
        metadata.description = data.get(Keys.General.DESCRIPTION)
        metadata.license = data.get(Keys.General.LICENSE)
        metadata.source_url = data.get(Keys.General.SOURCE_URL)
        metadata.source_hf_repo = data.get(Keys.General.SOURCE_HF_REPO)

        return metadata
