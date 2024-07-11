#!/usr/bin/env python3

import unittest
import gguf  # noqa: F401
from pathlib import Path


class TestMetadataMethod(unittest.TestCase):

    def test_id_to_title(self):
        self.assertEqual(gguf.Metadata.id_to_title("Mixtral-8x7B-Instruct-v0.1"), "Mixtral 8x7B Instruct v0.1")
        self.assertEqual(gguf.Metadata.id_to_title("Meta-Llama-3-8B"), "Meta Llama 3 8B")
        self.assertEqual(gguf.Metadata.id_to_title("hermes-2-pro-llama-3-8b-DPO"), "Hermes 2 Pro Llama 3 8b DPO")

    def test_get_model_id_components(self):
        self.assertEqual(gguf.Metadata.get_model_id_components("Mistral/Mixtral-8x7B-Instruct-v0.1"),
                         ('Mixtral-8x7B-Instruct-v0.1', "Mistral", 'Mixtral', 'Instruct', 'v0.1', '8x7B'))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral-8x7B-Instruct-v0.1"),
                         ('Mixtral-8x7B-Instruct-v0.1', None, 'Mixtral', 'Instruct', 'v0.1', '8x7B'))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral-8x7B-Instruct"),
                         ('Mixtral-8x7B-Instruct', None, 'Mixtral', 'Instruct', None, '8x7B'))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral-8x7B-v0.1"),
                         ('Mixtral-8x7B-v0.1', None, 'Mixtral', None, 'v0.1', '8x7B'))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral-8x7B"),
                         ('Mixtral-8x7B', None, 'Mixtral', None, None, '8x7B'))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral"),
                         ('Mixtral', None, 'Mixtral', None, None, None))
        self.assertEqual(gguf.Metadata.get_model_id_components("Mixtral-v0.1"),
                         ('Mixtral-v0.1', None, 'Mixtral', None, 'v0.1', None))
        self.assertEqual(gguf.Metadata.get_model_id_components("hermes-2-pro-llama-3-8b-DPO"),
                         ('hermes-2-pro-llama-3-8b-DPO', None, 'hermes-2-pro-llama-3', 'DPO', None, '8b'))
        self.assertEqual(gguf.Metadata.get_model_id_components("NousResearch/Meta-Llama-3-8B"),
                         ('Meta-Llama-3-8B', "NousResearch", 'Meta-Llama-3', None, None, "8B"))

    def test_apply_metadata_heuristic_from_model_card(self):
        model_card = {
            'tags': ['Llama-3', 'instruct', 'finetune', 'chatml', 'DPO', 'RLHF', 'gpt4', 'synthetic data', 'distillation', 'function calling', 'json mode', 'axolotl'],
            'model-index': [{'name': 'Hermes-2-Pro-Llama-3-8B', 'results': []}],
            'language': ['en'],
            'datasets': ['teknium/OpenHermes-2.5'],
            'widget': [{'example_title': 'Hermes 2 Pro', 'messages': [{'role': 'system', 'content': 'You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.'}, {'role': 'user', 'content': 'Write a short story about Goku discovering kirby has teamed up with Majin Buu to destroy the world.'}]}],
            'base_model': ["EmbeddedLLM/Mistral-7B-Merge-14-v0", "janai-hq/trinity-v1"]
        }
        got = gguf.Metadata.apply_metadata_heuristic(gguf.Metadata(), model_card, None, None)
        expect = gguf.Metadata(name=None, author=None, version=None, organization=None, finetune=None, basename=None, description=None, quantized_by=None, size_label=None, url=None, doi=None, uuid=None, repo_url=None, license=None, license_name=None, license_link=None, base_models=[{'name': 'Mistral 7B Merge 14 v0', 'organization': 'EmbeddedLLM', 'repo_url': 'https://huggingface.co/EmbeddedLLM/Mistral-7B-Merge-14-v0'}, {'name': 'Trinity v1', 'organization': 'Janai Hq', 'repo_url': 'https://huggingface.co/janai-hq/trinity-v1'}], tags=['Llama-3', 'instruct', 'finetune', 'chatml', 'DPO', 'RLHF', 'gpt4', 'synthetic data', 'distillation', 'function calling', 'json mode', 'axolotl'], languages=['en'], datasets=['teknium/OpenHermes-2.5'])
        self.assertEqual(got, expect)

    def test_apply_metadata_heuristic_from_hf_parameters(self):
        hf_params = {"_name_or_path": "./hermes-2-pro-llama-3-8b-DPO"}
        got = gguf.Metadata.apply_metadata_heuristic(gguf.Metadata(), None, hf_params, None)
        expect = gguf.Metadata(name='Hermes 2 Pro Llama 3 8b DPO', author=None, version=None, organization=None, finetune='DPO', basename='hermes-2-pro-llama-3', description=None, quantized_by=None, size_label='8b', url=None, doi=None, uuid=None, repo_url=None, license=None, license_name=None, license_link=None, base_models=None, tags=None, languages=None, datasets=None)
        self.assertEqual(got, expect)

    def test_apply_metadata_heuristic_from_model_dir(self):
        model_dir_path = Path("./hermes-2-pro-llama-3-8b-DPO")
        got = gguf.Metadata.apply_metadata_heuristic(gguf.Metadata(), None, None, model_dir_path)
        expect = gguf.Metadata(name='Hermes 2 Pro Llama 3 8b DPO', author=None, version=None, organization=None, finetune='DPO', basename='hermes-2-pro-llama-3', description=None, quantized_by=None, size_label='8b', url=None, doi=None, uuid=None, repo_url=None, license=None, license_name=None, license_link=None, base_models=None, tags=None, languages=None, datasets=None)
        self.assertEqual(got, expect)
