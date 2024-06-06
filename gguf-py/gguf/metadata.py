#!/usr/bin/env python3

from __future__ import annotations

import re
import json
import unittest
import frontmatter
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

if __name__ == '__main__':
    from constants import Keys
else:
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
    url: Optional[str] = None
    doi: Optional[str] = None
    uuid: Optional[str] = None
    hf_repo: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    license_name: Optional[str] = None
    license_link: Optional[str] = None
    source_url: Optional[str] = None
    source_doi: Optional[str] = None
    source_uuid: Optional[str] = None
    source_hf_repo: Optional[str] = None
    parameter_class_attribute: Optional[str] = None
    parents: Optional[list[dict]] = None
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

        model_card = Metadata.load_model_card(model_path)
        hf_params = Metadata.load_hf_parameters(model_path)

        # heuristics
        metadata = Metadata.apply_metadata_heuristic(metadata, model_card, hf_params, model_path)

        # Metadata Override File Provided
        # This is based on LLM_KV_NAMES mapping in llama.cpp
        metadata_override = Metadata.load_metadata_override(metadata_override_path)

        metadata.name                      = metadata_override.get(Keys.General.NAME                     ,  metadata.name                     ) # noqa: E202
        metadata.author                    = metadata_override.get(Keys.General.AUTHOR                   ,  metadata.author                   ) # noqa: E202
        metadata.version                   = metadata_override.get(Keys.General.VERSION                  ,  metadata.version                  ) # noqa: E202
        metadata.organization              = metadata_override.get(Keys.General.ORGANIZATION             ,  metadata.organization             ) # noqa: E202

        metadata.basename                  = metadata_override.get(Keys.General.BASENAME                 ,  metadata.basename                 ) # noqa: E202
        metadata.finetune                  = metadata_override.get(Keys.General.FINETUNE                 ,  metadata.finetune                 ) # noqa: E202
        metadata.description               = metadata_override.get(Keys.General.DESCRIPTION              ,  metadata.description              ) # noqa: E202
        metadata.quantized_by              = metadata_override.get(Keys.General.QUANTIZED_BY             ,  metadata.quantized_by             ) # noqa: E202
        metadata.parameter_class_attribute = metadata_override.get(Keys.General.PARAMETER_CLASS_ATTRIBUTE,  metadata.parameter_class_attribute) # noqa: E202

        metadata.license                   = metadata_override.get(Keys.General.LICENSE                  ,  metadata.license                  ) # noqa: E202
        metadata.license_name              = metadata_override.get(Keys.General.LICENSE_NAME             ,  metadata.license_name             ) # noqa: E202
        metadata.license_link              = metadata_override.get(Keys.General.LICENSE_LINK             ,  metadata.license_link             ) # noqa: E202

        metadata.url                       = metadata_override.get(Keys.General.URL                      ,  metadata.url                      ) # noqa: E202
        metadata.doi                       = metadata_override.get(Keys.General.DOI                      ,  metadata.doi                      ) # noqa: E202
        metadata.uuid                      = metadata_override.get(Keys.General.UUID                     ,  metadata.uuid                     ) # noqa: E202
        metadata.hf_repo                   = metadata_override.get(Keys.General.HF_REPO                  ,  metadata.hf_repo                  ) # noqa: E202

        metadata.source_url                = metadata_override.get(Keys.General.SOURCE_URL               ,  metadata.source_url               ) # noqa: E202
        metadata.source_doi                = metadata_override.get(Keys.General.SOURCE_DOI               ,  metadata.source_doi               ) # noqa: E202
        metadata.source_uuid               = metadata_override.get(Keys.General.SOURCE_UUID              ,  metadata.source_uuid              ) # noqa: E202
        metadata.source_hf_repo            = metadata_override.get(Keys.General.SOURCE_HF_REPO           ,  metadata.source_hf_repo           ) # noqa: E202

        metadata.parent_count              = metadata_override.get("general.parents"                     ,  metadata.parent_count             ) # noqa: E202

        metadata.tags                      = metadata_override.get(Keys.General.TAGS                     ,  metadata.tags                     ) # noqa: E202
        metadata.languages                 = metadata_override.get(Keys.General.LANGUAGES                ,  metadata.languages                ) # noqa: E202
        metadata.datasets                  = metadata_override.get(Keys.General.DATASETS                 ,  metadata.datasets                 ) # noqa: E202

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
    def load_hf_parameters(model_path: Optional[Path] = None) -> dict[str, object]:
        if model_path is None or not model_path.exists():
            return {}

        config_path = model_path / "config.json"

        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def id_to_title(string):
        # Convert capitalization into title form unless acronym or version number
        string = string.strip().replace('-', ' ')
        return ' '.join([w.title() if w.islower() and not re.match(r'^v\d+(?:\.\d+)*$', w) else w for w in string.split()])

    @staticmethod
    def get_model_id_components(model_id: Optional[str] = None) -> dict[str, object]:
        # Huggingface often store model id as '<org>/<model name>'
        # so let's parse it and apply some heuristics if possible for model name components

        if model_id is None:
            # model ID missing
            return None, None, None, None, None, None

        if ' ' in model_id:
            # model ID is actually a normal human sentence
            # which means its most likely a normal model name only
            # not part of the hugging face naming standard, but whatever
            return model_id, None, None, None, None, None

        if '/' in model_id:
            # model ID (huggingface style)
            org_component, model_full_name_component = model_id.split('/', 1)
        else:
            # model ID but missing org components
            org_component, model_full_name_component = None, model_id

        # Check if we erroneously matched against './' or '../' etc...
        if org_component is not None and org_component[0] == '.':
            org_component = None

        # Regular expression to extract model name components
        # Heuristic to match against cases such as 'Mixtral-8x7B-Instruct-v0.1' or 'Codestral-22B-v0.1'
        regex_match = re.compile(r'^(?P<basename>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))'
                                 r'(?:-(?P<parameter_class_attribute>(?:\d+x)?\d+[A-Za-z]+)(?:-(?P<finetune>[A-Za-z0-9\s-]+))?)?'
                                 r'(?:-(?P<version>v\d+(?:\.\d+)*))?$').match(model_full_name_component)

        if not regex_match:
            return model_full_name_component, org_component, None, None, None, None

        components = regex_match.groupdict()
        basename = components.get("basename")
        finetune = components.get("finetune")
        version = components.get("version")
        parameter_class_attribute = components.get("parameter_class_attribute")

        return model_full_name_component, org_component, basename, finetune, version, parameter_class_attribute

    @staticmethod
    def apply_metadata_heuristic(metadata: Metadata, model_card: Optional[dict] = None, hf_params: Optional[dict] = None, model_path: Optional[Path] = None) -> Metadata:
        # Reference Model Card Metadata: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
        found_base_model = False

        # Model Card Heuristics
        ########################
        if model_card is not None:

            if "model_name" in model_card:
                # Not part of huggingface model card standard but notice some model creator using it
                # such as TheBloke who would encode 'Mixtral 8X7B Instruct v0.1' into model_name
                metadata.name = model_card.get("model_name")

            if "base_model" in model_card and isinstance(model_card["base_model"], str) and not found_base_model:
                # Check if string. We cannot handle lists as that is too ambagious
                # Example: stabilityai/stable-diffusion-xl-base-1.0. Can also be a list (for merges)
                model_id = model_card.get("base_model")
                model_full_name_component, org_component, basename, finetune, version, parameter_class_attribute = Metadata.get_model_id_components(model_id)
                if metadata.name is None and model_full_name_component is not None:
                    metadata.name = Metadata.id_to_title(model_full_name_component)
                if metadata.organization is None and org_component is not None:
                    metadata.organization = Metadata.id_to_title(org_component)
                if metadata.basename is None and basename is not None:
                    metadata.basename = basename
                if metadata.finetune is None and finetune is not None:
                    metadata.finetune = finetune
                if metadata.version is None and version is not None:
                    metadata.version = version
                if metadata.parameter_class_attribute is None and parameter_class_attribute is not None:
                    metadata.parameter_class_attribute = parameter_class_attribute
                if metadata.source_url is None and org_component is not None and model_full_name_component is not None:
                    metadata.source_url = f"https://huggingface.co/{org_component}/{model_full_name_component}"
                if metadata.source_hf_repo is None and org_component is not None and model_full_name_component is not None:
                    metadata.source_hf_repo = f"{org_component}/{model_full_name_component}"

                found_base_model = True

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
                metadata.tags = model_card.get("tags", None)
            if metadata.languages is None:
                metadata.languages = model_card.get("language", model_card.get("languages", None))
            if metadata.datasets is None:
                metadata.datasets = model_card.get("datasets", model_card.get("dataset", None))

        # Hugging Face Parameter Heuristics
        ####################################

        if hf_params is not None:
            hf_name_or_path = hf_params.get("_name_or_path")

            if hf_name_or_path is not None and hf_name_or_path.count('/') <= 1 and not found_base_model:
                # Use _name_or_path only if its actually a model name and not some computer path
                # e.g. 'meta-llama/Llama-2-7b-hf'
                model_id = hf_name_or_path
                model_full_name_component, org_component, basename, finetune, version, parameter_class_attribute = Metadata.get_model_id_components(model_id)
                if metadata.name is None and model_full_name_component is not None:
                    metadata.name = Metadata.id_to_title(model_full_name_component)
                if metadata.organization is None and org_component is not None:
                    metadata.organization = Metadata.id_to_title(org_component)
                if metadata.basename is None and basename is not None:
                    metadata.basename = basename
                if metadata.finetune is None and finetune is not None:
                    metadata.finetune = finetune
                if metadata.version is None and version is not None:
                    metadata.version = version
                if metadata.parameter_class_attribute is None and parameter_class_attribute is not None:
                    metadata.parameter_class_attribute = parameter_class_attribute
                if metadata.source_url is None and org_component is not None and model_full_name_component is not None:
                    metadata.source_url = f"https://huggingface.co/{org_component}/{model_full_name_component}"
                if metadata.source_hf_repo is None and org_component is not None and model_full_name_component is not None:
                    metadata.source_hf_repo = f"{org_component}/{model_full_name_component}"

        # Directory Folder Name Fallback Heuristics
        ############################################
        if model_path is not None:
            model_id = model_path.name
            model_full_name_component, org_component, basename, finetune, version, parameter_class_attribute = Metadata.get_model_id_components(model_id)
            if metadata.name is None and model_full_name_component is not None:
                metadata.name = Metadata.id_to_title(model_full_name_component)
            if metadata.organization is None and org_component is not None:
                metadata.organization = Metadata.id_to_title(org_component)
            if metadata.basename is None and basename is not None:
                metadata.basename = basename
            if metadata.finetune is None and finetune is not None:
                metadata.finetune = finetune
            if metadata.version is None and version is not None:
                metadata.version = version
            if metadata.parameter_class_attribute is None and parameter_class_attribute is not None:
                metadata.parameter_class_attribute = parameter_class_attribute
            if metadata.source_url is None and org_component is not None and model_full_name_component is not None:
                metadata.source_url = f"https://huggingface.co/{org_component}/{model_full_name_component}"
            if metadata.source_hf_repo is None and org_component is not None and model_full_name_component is not None:
                metadata.source_hf_repo = f"{org_component}/{model_full_name_component}"

        return metadata


class TestStringMethods(unittest.TestCase):

    def test_get_model_id_components(self):
        self.assertEqual(Metadata.get_model_id_components("Mistral/Mixtral-8x7B-Instruct-v0.1"),
                         ('Mixtral-8x7B-Instruct-v0.1', "Mistral", 'Mixtral', 'Instruct', 'v0.1', '8x7B'))
        self.assertEqual(Metadata.get_model_id_components("Mixtral-8x7B-Instruct-v0.1"),
                         ('Mixtral-8x7B-Instruct-v0.1', None, 'Mixtral', 'Instruct', 'v0.1', '8x7B'))
        self.assertEqual(Metadata.get_model_id_components("Mixtral-8x7B-Instruct"),
                         ('Mixtral-8x7B-Instruct', None, 'Mixtral', 'Instruct', None, '8x7B'))
        self.assertEqual(Metadata.get_model_id_components("Mixtral-8x7B-v0.1"),
                         ('Mixtral-8x7B-v0.1', None, 'Mixtral', None, 'v0.1', '8x7B'))
        self.assertEqual(Metadata.get_model_id_components("Mixtral-8x7B"),
                         ('Mixtral-8x7B', None, 'Mixtral', None, None, '8x7B'))
        self.assertEqual(Metadata.get_model_id_components("Mixtral"),
                         ('Mixtral', None, 'Mixtral', None, None, None))
        self.assertEqual(Metadata.get_model_id_components("Mixtral-v0.1"),
                         ('Mixtral-v0.1', None, 'Mixtral', None, 'v0.1', None))
        self.assertEqual(Metadata.get_model_id_components("hermes-2-pro-llama-3-8b-DPO"),
                         ('hermes-2-pro-llama-3-8b-DPO', None, 'hermes-2-pro-llama-3', 'DPO', None, '8b'))
        self.assertEqual(Metadata.get_model_id_components("NousResearch/Meta-Llama-3-8B"),
                         ('Meta-Llama-3-8B', "NousResearch", 'Meta-Llama-3', None, None, "8B"))

    def test_apply_metadata_heuristic_from_model_card(self):
        # Source: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/README.md
        model_card = {
            'base_model': 'NousResearch/Meta-Llama-3-8B',
            'tags': ['Llama-3', 'instruct', 'finetune', 'chatml', 'DPO', 'RLHF', 'gpt4', 'synthetic data', 'distillation', 'function calling', 'json mode', 'axolotl'],
            'model-index': [{'name': 'Hermes-2-Pro-Llama-3-8B', 'results': []}],
            'language': ['en'],
            'datasets': ['teknium/OpenHermes-2.5'],
            'widget': [{'example_title': 'Hermes 2 Pro', 'messages': [{'role': 'system', 'content': 'You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.'}, {'role': 'user', 'content': 'Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.'}]}]
        }
        expected = Metadata(name='Meta Llama 3 8B', basename='Meta-Llama-3', finetune=None, author=None, quantized_by=None, organization='NousResearch', version=None, url=None, doi=None, uuid=None, hf_repo=None, description=None, license=None, license_name=None, license_link=None, source_url='https://huggingface.co/NousResearch/Meta-Llama-3-8B', source_doi=None, source_uuid=None, source_hf_repo='NousResearch/Meta-Llama-3-8B', parameter_class_attribute='8B', parents=None, tags=['Llama-3', 'instruct', 'finetune', 'chatml', 'DPO', 'RLHF', 'gpt4', 'synthetic data', 'distillation', 'function calling', 'json mode', 'axolotl'], languages=['en'], datasets=['teknium/OpenHermes-2.5'])

        got = Metadata.apply_metadata_heuristic(Metadata(), model_card, None, None)

        self.assertEqual(got, expected)

    def test_apply_metadata_heuristic_from_hf_parameters(self):
        # Source: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/config.json
        hf_params = {"_name_or_path": "./hermes-2-pro-llama-3-8b-DPO"}
        expected = Metadata(name='Hermes 2 Pro Llama 3 8B DPO', basename='hermes-2-pro-llama-3', finetune='DPO', author=None, quantized_by=None, organization=None, version=None, url=None, doi=None, uuid=None, hf_repo=None, description=None, license=None, license_name=None, license_link=None, source_url=None, source_doi=None, source_uuid=None, source_hf_repo=None, parameter_class_attribute='8b', parents=None, tags=None, languages=None, datasets=None)
        got = Metadata.apply_metadata_heuristic(Metadata(), None, hf_params, None)
        self.assertEqual(got, expected)

    def test_apply_metadata_heuristic_from_model_dir(self):
        # Source: https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/config.json
        model_dir_path = Path("./hermes-2-pro-llama-3-8b-DPO")
        expected = Metadata(name='Hermes 2 Pro Llama 3 8B DPO', basename='hermes-2-pro-llama-3', finetune='DPO', author=None, quantized_by=None, organization=None, version=None, url=None, doi=None, uuid=None, hf_repo=None, description=None, license=None, license_name=None, license_link=None, source_url=None, source_doi=None, source_uuid=None, source_hf_repo=None, parameter_class_attribute='8b', parents=None, tags=None, languages=None, datasets=None)
        got = Metadata.apply_metadata_heuristic(Metadata(), None, None, model_dir_path)
        self.assertEqual(got, expected)


if __name__ == '__main__':
    unittest.main()
