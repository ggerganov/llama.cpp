import json
import logging
import os
import pathlib
from enum import IntEnum, auto
from hashlib import sha256

import requests
from transformers import AutoTokenizer

from .constants import MODEL_ARCH


class LLaMaVocabType(IntEnum):
    NON = auto()  # For models without vocab
    SPM = auto()  # SentencePiece LLaMa tokenizer
    BPE = auto()  # BytePair GPT-2 tokenizer
    WPM = auto()  # WordPiece BERT tokenizer


class LLaMaModelType(IntEnum):
    PTH = auto()  # PyTorch
    SFT = auto()  # SafeTensor


# NOTE:
#   - Repository paths are required
#   - Allow the user to specify the tokenizer model type themselves
#   - Use architecture types because they are explicitly defined
#   - Possible tokenizer model types are: SentencePiece, WordPiece, or BytePair
MODELS = (
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.SPM, "repo": "meta-llama/Llama-2-7b-hf", },
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.BPE, "repo": "meta-llama/Meta-Llama-3-8B", },
    {"model_arch": MODEL_ARCH.PHI3, "vocab_type": LLaMaVocabType.SPM, "repo": "microsoft/Phi-3-mini-4k-instruct", },
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.BPE, "repo": "deepseek-ai/deepseek-llm-7b-base", },
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.BPE, "repo": "deepseek-ai/deepseek-coder-6.7b-base", },
    {"model_arch": MODEL_ARCH.FALCON, "vocab_type": LLaMaVocabType.BPE, "repo": "tiiuae/falcon-7b", },
    {"model_arch": MODEL_ARCH.BERT, "vocab_type": LLaMaVocabType.WPM, "repo": "BAAI/bge-small-en-v1.5", },
    {"model_arch": MODEL_ARCH.MPT, "vocab_type": LLaMaVocabType.BPE, "repo": "mosaicml/mpt-7b", },
    {"model_arch": MODEL_ARCH.STARCODER2, "vocab_type": LLaMaVocabType.BPE, "repo": "bigcode/starcoder2-3b", },
    {"model_arch": MODEL_ARCH.GPT2, "vocab_type": LLaMaVocabType.BPE, "repo": "openai-community/gpt2", },
    {"model_arch": MODEL_ARCH.REFACT, "vocab_type": LLaMaVocabType.BPE, "repo": "smallcloudai/Refact-1_6-base", },
    {"model_arch": MODEL_ARCH.COMMAND_R, "vocab_type": LLaMaVocabType.BPE, "repo": "CohereForAI/c4ai-command-r-v01", },
    {"model_arch": MODEL_ARCH.QWEN2, "vocab_type": LLaMaVocabType.BPE, "repo": "Qwen/Qwen1.5-7B", },
    {"model_arch": MODEL_ARCH.OLMO, "vocab_type": LLaMaVocabType.BPE, "repo": "allenai/OLMo-1.7-7B-hf", },
    {"model_arch": MODEL_ARCH.DBRX, "vocab_type": LLaMaVocabType.BPE, "repo": "databricks/dbrx-base", },
    {"model_arch": MODEL_ARCH.JINA_BERT_V2, "vocab_type": LLaMaVocabType.WPM, "repo": "jinaai/jina-embeddings-v2-base-en", },
    {"model_arch": MODEL_ARCH.JINA_BERT_V2, "vocab_type": LLaMaVocabType.BPE, "repo": "jinaai/jina-embeddings-v2-base-es", },
    {"model_arch": MODEL_ARCH.JINA_BERT_V2, "vocab_type": LLaMaVocabType.BPE, "repo": "jinaai/jina-embeddings-v2-base-de", },
    {"model_arch": MODEL_ARCH.PHI2, "vocab_type": LLaMaVocabType.BPE, "repo": "microsoft/phi-1", },
    {"model_arch": MODEL_ARCH.STABLELM, "vocab_type": LLaMaVocabType.BPE, "repo": "stabilityai/stablelm-2-zephyr-1_6b", },
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.SPM, "repo": "mistralai/Mistral-7B-Instruct-v0.2", },
    {"model_arch": MODEL_ARCH.LLAMA, "vocab_type": LLaMaVocabType.SPM, "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1", },
)


class HFHubRequest:
    def __init__(self, auth_token: None | str, logger: None | logging.Logger):
        # Set headers if authentication is available
        if auth_token is None:
            self._headers = None
        else:
            self._headers = {"Authorization": f"Bearer {auth_token}"}

        # Set the logger
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Persist across requests
        self._session = requests.Session()

        # This is read-only
        self._base_url = "https://huggingface.co"

    @property
    def headers(self) -> str:
        return self._headers

    @property
    def save_path(self) -> pathlib.Path:
        return self._save_path

    @property
    def session(self) -> requests.Session:
        return self._session

    @property
    def base_url(self) -> str:
        return self._base_url

    def write_file(self, content: bytes, filepath: pathlib.Path) -> None:
        with open(filepath, 'wb') as f:
            f.write(content)
        self.logger.info(f"Wrote {len(content)} bytes to {filepath} successfully")

    def resolve_url(self, repo: str, filename: str) -> str:
        return f"{self._base_url}/{repo}/resolve/main/{filename}"

    def download_file(self, url: str) -> requests.Response:
        response = self._session.get(url, headers=self.headers)
        self.logger.info(f"Response status was {response.status_code}")
        response.raise_for_status()
        return response


class HFHubBase:
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        if model_path is None:
            self._model_path = pathlib.Path("models")
        elif isinstance(model_path, str):
            self._model_path = pathlib.Path(model_path)
        else:
            self._model_path = model_path

        # Set the logger
        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger(__name__)
        self.logger = logger

        self._hub = HFHubRequest(auth_token, logger)
        self._models = list(MODELS)

    @property
    def hub(self) -> HFHubRequest:
        return self._hub

    @property
    def models(self) -> list[dict[str, object]]:
        return self._models

    @property
    def model_path(self) -> pathlib.Path:
        return self._model_path

    @model_path.setter
    def model_path(self, value: pathlib.Path):
        self._model_path = value


class HFVocabRequest(HFHubBase):
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        super().__init__(model_path, auth_token, logger)

    @property
    def tokenizer_type(self) -> LLaMaVocabType:
        return LLaMaVocabType

    def resolve_filenames(self, tokt: LLaMaVocabType) -> tuple[str]:
        filenames = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        if tokt == self.tokenizer_type.SPM:
            filenames.append("tokenizer.model")
        return tuple(filenames)

    def resolve_tokenizer_model(
        self,
        filename: str,
        filepath: pathlib.Path,
        model: dict[str, object]
    ) -> None:
        try:  # NOTE: Do not use bare exceptions! They mask issues!
            resolve_url = self.hub.resolve_url(model['repo'], filename)
            response = self.hub.download_file(resolve_url)
            self.hub.write_file(response.content, filepath)
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Failed to download tokenizer {model['repo']}: {e}")

    def download_models(self) -> None:
        for model in self.models:
            os.makedirs(f"{self.model_path}/{model['repo']}", exist_ok=True)
            filenames = self.resolve_filenames(model['tokt'])
            for filename in filenames:
                filepath = pathlib.Path(f"{self.model_path}/{model['repo']}/{filename}")
                if filepath.is_file():
                    self.logger.info(f"skipped pre-existing tokenizer {model['repo']} in {filepath}")
                    continue
                self.resolve_tokenizer_model(filename, filepath, model)

    def generate_checksums(self) -> None:
        checksums = []
        for model in self.models:
            mapping = {}
            filepath = f"{self.model_path}/{model['repo']}"

            try:
                tokenizer = AutoTokenizer.from_pretrained(filepath, trust_remote=True)
            except OSError as e:
                self.logger.error(f"Failed to hash tokenizer {model['repo']}: {e}")
                continue

            mapping.update(model)
            mapping['checksum'] = sha256(str(tokenizer.vocab).encode()).hexdigest()
            self.logger.info(f"Hashed {mapping['repo']} as {mapping['checksum']}")
            checksums.append(mapping)

        with open(f"{self.model_path}/checksums.json", mode="w") as file:
            json.dump(checksums, file)

    def log_pre_tokenizer_info(self) -> None:
        for model in self.models:
            try:
                with open(f"{self.model_path}/{model['repo']}/tokenizer.json", "r", encoding="utf-8") as f:
                    self.logger.info(f"Start: {model['repo']}")
                    cfg = json.load(f)
                    self.logger.info(f"normalizer: {json.dumps(cfg['normalizer'], indent=4)}")
                    self.logger.info(f"pre_tokenizer: {json.dumps(cfg['pre_tokenizer'], indent=4)}")
                    if "type" in cfg["model"]:
                        self.logger.info(f"type: {json.dumps(cfg['model']['type'])}")
                    if "ignore_merges" in cfg["model"]:
                        self.logger.info(f"ignore_merges: {json.dumps(cfg['model']['ignore_merges'], indent=4)}")
                self.logger.info(f"End: {model['repo']}")
            except FileNotFoundError as e:
                self.logger.error(f"Failed to log tokenizer {model['repo']}: {e}")


# TODO:
class HFModelRequest(HFHubBase):
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        super().__init__(model_path, auth_token, logger)

    @property
    def model_type(self) -> LLaMaModelType:
        return LLaMaModelType
