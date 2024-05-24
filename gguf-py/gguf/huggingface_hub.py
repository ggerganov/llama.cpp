import json
import logging
import os
import pathlib
from hashlib import sha256

import requests
from transformers import AutoTokenizer

from .constants import (
    GPT_PRE_TOKENIZER_DEFAULT,
    HF_TOKENIZER_BPE_FILES,
    HF_TOKENIZER_SPM_FILES,
    MODEL_FILE_TYPE_NAMES,
    VOCAB_TYPE_NAMES,
    ModelFileType,
    VocabType,
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

    def write_file(self, content: bytes, file_path: pathlib.Path) -> None:
        with open(file_path, 'wb') as f:
            f.write(content)
        self.logger.info(f"Wrote {len(content)} bytes to {file_path} successfully")

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

    @property
    def hub(self) -> HFHubRequest:
        return self._hub

    @property
    def model_path(self) -> pathlib.Path:
        return self._model_path

    @model_path.setter
    def model_path(self, value: pathlib.Path):
        self._model_path = value


class HFVocabRequest(HFHubBase):
    def __init__(
        self,
        auth_token: str,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger
    ):
        super().__init__(model_path, auth_token, logger)

    @property
    def tokenizer_type(self) -> VocabType:
        return VocabType

    @property
    def tokenizer_path(self) -> pathlib.Path:
        return self.model_path / "tokenizer.json"

    def get_vocab_name(self, vocab_type: VocabType) -> str:
        return VOCAB_TYPE_NAMES.get(vocab_type)

    def get_vocab_enum(self, vocab_name: str) -> VocabType:
        return {
            "SPM": VocabType.SPM,
            "BPE": VocabType.BPE,
            "WPM": VocabType.WPM,
        }.get(vocab_name, VocabType.NON)

    def get_vocab_filenames(self, vocab_type: VocabType) -> tuple[str]:
        if vocab_type == self.tokenizer_type.SPM:
            return HF_TOKENIZER_SPM_FILES
        # NOTE: WPM and BPE are equivalent
        return HF_TOKENIZER_BPE_FILES

    def get_vocab_file(
        self, model_repo: str, file_name: str, file_path: pathlib.Path,
    ) -> bool:
        # NOTE: Do not use bare exceptions! They mask issues!
        # Allow the exception to occur or handle it explicitly.
        resolve_url = self.hub.resolve_url(model_repo, file_name)
        response = self.hub.download_file(resolve_url)
        self.hub.write_file(response.content, file_path)
        self.logger.info(f"Downloaded tokenizer {file_name} from {model_repo}")

    def get_all_vocab_files(self, model_repo: str, vocab_type: VocabType) -> None:
        vocab_list = self.get_vocab_filenames(vocab_type)
        for vocab_file in vocab_list:
            dir_path = self.model_path / model_repo
            file_path = dir_path / vocab_file
            os.makedirs(dir_path, exist_ok=True)
            self.get_vocab_file(model_repo, vocab_file, file_path)

    def get_normalizer(self) -> None | dict[str, object]:
        with open(self.tokenizer_path, mode="r") as file:
            tokenizer_json = json.load(file)
        return tokenizer_json.get("normalizer")

    def get_pre_tokenizer(self) -> None | dict[str, object]:
        with open(self.tokenizer_path, mode="r") as file:
            tokenizer_json = json.load(file)
        return tokenizer_json.get("pre_tokenizer")

    def generate_checksum(self) -> None:
        checksums = []
        for model in self.models:
            mapping = {}
            file_path = f"{self.model_path}/{model['repo']}"

            try:
                tokenizer = AutoTokenizer.from_pretrained(file_path, trust_remote=True)
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
    def model_type(self) -> ModelFileType:
        return ModelFileType
