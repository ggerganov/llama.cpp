from __future__ import annotations

import re
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
    quantized_by: Optional[str] = None
    organization: Optional[str] = None
    version: Optional[str] = None
    base_version: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_link: Optional[str] = None
    source_url: Optional[str] = None
    source_hf_repo: Optional[str] = None
    parameter_weight_class: Optional[str] = None
    tags: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    datasets: Optional[list[str]] = None

    @staticmethod
    def load(metadata_override_path: Optional[Path] = None, model_path: Optional[Path] = None, model_name: Optional[str] = None) -> Metadata:
        # This grabs as many contextual authorship metadata as possible from the model repository
        # making any conversion as required to match the gguf kv store metadata format
        # as well as giving users the ability to override any authorship metadata that may be incorrect

        # Create a new Metadata instance
        metadata = Metadata()

        # load huggingface model card if available
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        model_card = Metadata.load_model_card(model_path)

        if "model_name" in model_card:
            # Not part of huggingface model card standard but notice some model creator using it
            # such as TheBloke who would encode 'Mixtral 8X7B Instruct v0.1' into model_name
            metadata.name = model_card.get("model_name")

        if "base_model" in model_card:
            # Not part of huggingface model card standard but notice some model creator using it
            # such as TheBloke who would encode 'mistralai/Mixtral-8x7B-Instruct-v0.1' into base_model
            model_id = model_card.get("base_model")
            model_name_normal, organization_name, base_name, fine_tune, version_string, parameter_weight_class = Metadata.get_model_name_components(model_id)

            if metadata.name is None and model_name_normal is not None:
                metadata.name = model_name_normal
            if metadata.organization is None and organization_name is not None:
                metadata.organization = organization_name
            if metadata.basename is None and base_name is not None:
                metadata.basename = base_name
            if metadata.finetune is None and fine_tune is not None:
                metadata.finetune = fine_tune
            if metadata.version is None and version_string is not None:
                metadata.version = version_string
            if metadata.parameter_weight_class is None and parameter_weight_class is not None:
                metadata.parameter_weight_class = parameter_weight_class

        if "model-index" in model_card and len(model_card["model_name"]) == 1 and "name" in model_card["model_name"][0]:
            # This is a model index which has model id that can be extracted into organization and model name
            # if so then we can safely extract organization and name
            # (This is a safe choice in case there is multiple models in one repo in the future)
            model_id = model_card["model-index"][0].get("name")
            model_name_normal, organization_name, base_name, fine_tune, version_string, parameter_weight_class = Metadata.get_model_name_components(model_id)

            if metadata.name is None and model_name_normal is not None:
                metadata.name = model_name_normal
            if metadata.organization is None and organization_name is not None:
                metadata.organization = organization_name
            if metadata.basename is None and base_name is not None:
                metadata.basename = base_name
            if metadata.finetune is None and fine_tune is not None:
                metadata.finetune = fine_tune
            if metadata.version is None and version_string is not None:
                metadata.version = version_string
            if metadata.parameter_weight_class is None and parameter_weight_class is not None:
                metadata.parameter_weight_class = parameter_weight_class

        if metadata.quantized_by is None:
            # Not part of hugging face model card standard, but is used by TheBloke to credit them self for quantizing 3rd party models
            metadata.quantized_by = model_card.get("quantized_by")
        if metadata.license is None:
            metadata.license = model_card.get("license")
        if metadata.license_name is None:
            metadata.license_name = model_card.get("license_name")
        if metadata.license_link is None:
            metadata.license_link = model_card.get("license_link")
        if metadata.author is None:
            # non huggingface model card standard but notice some model creator using it
            metadata.author = model_card.get("model_creator")
        if metadata.tags is None:
            metadata.tags = model_card.get("tags", [])
        if metadata.languages is None:
            metadata.languages = model_card.get("language", model_card.get("languages", []))
        if metadata.datasets is None:
            metadata.datasets = model_card.get("datasets", model_card.get("dataset", []))

        # load huggingface parameters if available
        hf_params = Metadata.load_huggingface_parameters(model_path)

        hf_name_or_path = hf_params.get("_name_or_path")
        if hf_name_or_path is not None and Metadata.is_model_id(hf_name_or_path):
            # Use _name_or_path only if its actually a model name and not some computer path
            # e.g. 'meta-llama/Llama-2-7b-hf'
            model_name_normal, organization_name, base_name, fine_tune, version_string, parameter_weight_class = Metadata.get_model_name_components(hf_name_or_path)
            if metadata.name is None and model_name_normal is not None:
                metadata.name = model_name_normal
            if metadata.organization is None and organization_name is not None:
                metadata.organization = organization_name
            if metadata.basename is None and base_name is not None:
                metadata.basename = base_name
            if metadata.finetune is None and fine_tune is not None:
                metadata.finetune = fine_tune
            if metadata.version is None and version_string is not None:
                metadata.version = version_string
            if metadata.parameter_weight_class is None and parameter_weight_class is not None:
                metadata.parameter_weight_class = parameter_weight_class
            if metadata.source_hf_repo is None and not Metadata.is_model_name_only(hf_name_or_path):
                # Can't just have the model name as the source hf repo as a link to the huggingface website needs the org name and the model name
                metadata.source_hf_repo = "https://huggingface.co/{hf_name_or_path}"

        # Use Directory Folder Name As Fallback Name
        if metadata.name is None:
            if model_path is not None and model_path.exists():
                metadata.name = model_path.name

        # Metadata Override File Provided
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata_override = Metadata.load_metadata_override(metadata_override_path)
        metadata.name                   = metadata_override.get(Keys.General.NAME                  ,  metadata.name                  ) # noqa: E202
        metadata.basename               = metadata_override.get(Keys.General.BASENAME              ,  metadata.basename              ) # noqa: E202
        metadata.finetune               = metadata_override.get(Keys.General.FINETUNE              ,  metadata.finetune              ) # noqa: E202
        metadata.author                 = metadata_override.get(Keys.General.AUTHOR                ,  metadata.author                ) # noqa: E202
        metadata.quantized_by           = metadata_override.get(Keys.General.QUANTIZED_BY          ,  metadata.quantized_by          ) # noqa: E202
        metadata.organization           = metadata_override.get(Keys.General.ORGANIZATION          ,  metadata.organization          ) # noqa: E202
        metadata.version                = metadata_override.get(Keys.General.VERSION               ,  metadata.version               ) # noqa: E202
        metadata.base_version           = metadata_override.get(Keys.General.BASE_VERSION          ,  metadata.base_version          ) # noqa: E202
        metadata.url                    = metadata_override.get(Keys.General.URL                   ,  metadata.url                   ) # noqa: E202
        metadata.description            = metadata_override.get(Keys.General.DESCRIPTION           ,  metadata.description           ) # noqa: E202
        metadata.license                = metadata_override.get(Keys.General.LICENSE               ,  metadata.license               ) # noqa: E202
        metadata.license_name           = metadata_override.get(Keys.General.LICENSE_NAME          ,  metadata.license_name          ) # noqa: E202
        metadata.license_link           = metadata_override.get(Keys.General.LICENSE_LINK          ,  metadata.license_link          ) # noqa: E202
        metadata.source_url             = metadata_override.get(Keys.General.SOURCE_URL            ,  metadata.source_url            ) # noqa: E202
        metadata.source_hf_repo         = metadata_override.get(Keys.General.SOURCE_HF_REPO        ,  metadata.source_hf_repo        ) # noqa: E202
        metadata.parameter_weight_class = metadata_override.get(Keys.General.PARAMETER_WEIGHT_CLASS,  metadata.parameter_weight_class) # noqa: E202
        metadata.tags                   = metadata_override.get(Keys.General.TAGS                  ,  metadata.tags                  ) # noqa: E202
        metadata.languages              = metadata_override.get(Keys.General.LANGUAGES             ,  metadata.languages             ) # noqa: E202
        metadata.datasets               = metadata_override.get(Keys.General.DATASETS              ,  metadata.datasets              ) # noqa: E202

        # Direct Metadata Override (via direct cli argument)
        if model_name is not None:
            metadata.name = model_name

        return metadata

    @staticmethod
    def load_metadata_override(metadata_override_path: Optional[Path] = None) -> dict[str, object]:
        if metadata_override_path is None or not metadata_override_path.exists():
            return {}

        with open(metadata_override_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_model_card(model_path: Optional[Path] = None) -> dict[str, object]:
        if model_path is None or not model_path.exists():
            return {}

        model_card_path = model_path / "README.md"

        if not model_card_path.exists():
            return {}

        with open(model_card_path, "r", encoding="utf-8") as f:
            return frontmatter.load(f)

    @staticmethod
    def load_huggingface_parameters(model_path: Optional[Path] = None) -> dict[str, object]:
        if model_path is None or not model_path.exists():
            return {}

        config_path = model_path / "config.json"

        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def is_model_id(name_or_path: Optional[str] = None) -> bool:
        # Return True if the string has 1 or 0 slashes, indicating a model id
        # Created specifically because of _name_or_path in hugging face parameter
        if name_or_path is None:
            return False
        return name_or_path.count('/') <= 1

    @staticmethod
    def is_model_name_only(name_or_path: Optional[str] = None) -> bool:
        # Return True if the string has 0 slashes, indicating a model name only model id
        # Created specifically because of _name_or_path in hugging face parameter
        if name_or_path is None:
            return False
        return name_or_path.count('/') == 0

    @staticmethod
    def get_model_name_components(model_identifier: Optional[str] = None) -> dict[str, object]:
        # Huggingface often store model id

        if model_identifier is None:
            # model ID missing
            return None, None, None, None, None, None

        if ' ' in model_identifier:
            # model ID is actually a normal human sentence
            # which means its most likely a normal model name only
            # not part of the hugging face naming standard, but whatever
            return model_identifier, None, None, None, None, None

        if '/' in model_identifier:
            # model ID (huggingface style)
            organization, model = model_identifier.split('/', 1)
        else:
            # model ID but missing org components
            model = model_identifier
            organization = None

        # Apply formatting to organization and model_name
        # 'stable-diffusion-xl-base-1.0' --> 'Stable Diffusion Xl Base 1.0'

        organization_name = organization.strip().replace('-', ' ').title() if organization is not None else None
        model_name_normal = model.strip().replace('-', ' ').title() if model is not None else None

        # Regular expression to extract model name components
        # Heuristic to match against cases such as 'Mixtral-8x7B-Instruct-v0.1' or 'Codestral-22B-v0.1'

        regex_match = re.compile(r'^(?P<base_name>[A-Za-z0-9\s]*(?:(?:-[A-Za-z\s][A-Za-z0-9\s]*)*))'
                                 r'(?:-(?P<parameter_weight_class>(?:\d+x)?\d+[A-Za-z]+))?'
                                 r'(?:-(?P<fine_tune>[A-Za-z0-9\s-]+))?'
                                 r'(?:-(?P<version_string>v\d+(?:\.\d+)*))?$').match(model)

        if not regex_match:
            return model_name_normal, organization_name, None, None, None, None

        components = regex_match.groupdict()
        base_name = components.get("base_name")
        fine_tune = components.get("fine_tune")
        version_string = components.get("version_string")
        parameter_weight_class = components.get("parameter_weight_class")

        return model_name_normal, organization_name, base_name, fine_tune, version_string, parameter_weight_class
