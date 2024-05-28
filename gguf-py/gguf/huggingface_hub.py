import json
import logging
import os
import pathlib
from hashlib import sha256

import requests
from huggingface_hub import login, model_info
from sentencepiece import SentencePieceProcessor

from .constants import (
    GPT_PRE_TOKENIZER_DEFAULT,
    HF_TOKENIZER_BPE_FILES,
    HF_TOKENIZER_SPM_FILES,
    ModelFileExtension,
    NormalizerType,
    PreTokenizerType,
    VocabType,
)


class HFHubBase:
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger,
    ):
        # Set the model path
        if model_path is None:
            model_path = "models"
        self._model_path = model_path

        # Set the logger
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    @property
    def model_path(self) -> pathlib.Path:
        return pathlib.Path(self._model_path)

    @model_path.setter
    def model_path(self, value: pathlib.Path):
        self._model_path = value

    def write_file(self, content: bytes, file_path: pathlib.Path) -> None:
        with open(file_path, "wb") as file:
            file.write(content)
        self.logger.info(f"Wrote {len(content)} bytes to {file_path} successfully")


class HFHubRequest(HFHubBase):
    def __init__(
        self,
        auth_token: None | str,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger,
    ):
        super().__init__(model_path, logger)

        # Set headers if authentication is available
        if auth_token is None:
            self._headers = None
        else:
            self._headers = {"Authorization": f"Bearer {auth_token}"}

        # Persist across requests
        self._session = requests.Session()

        # This is read-only
        self._base_url = "https://huggingface.co"

        # NOTE: Required for getting model_info
        login(auth_token, add_to_git_credential=True)

    @property
    def headers(self) -> str:
        return self._headers

    @property
    def session(self) -> requests.Session:
        return self._session

    @property
    def base_url(self) -> str:
        return self._base_url

    @staticmethod
    def list_remote_files(model_repo: str) -> list[str]:
        # NOTE: Request repository metadata to extract remote filenames
        return [x.rfilename for x in model_info(model_repo).siblings]

    def list_filtered_remote_files(
        self, model_repo: str, file_extension: ModelFileExtension
    ) -> list[str]:
        model_files = []
        self.logger.info(f"Repo:{model_repo}")
        self.logger.debug(f"FileExtension:{file_extension.value}")
        for filename in HFHubRequest.list_remote_files(model_repo):
            suffix = pathlib.Path(filename).suffix
            self.logger.debug(f"Suffix: {suffix}")
            if suffix == file_extension.value:
                self.logger.info(f"File: {filename}")
                model_files.append(filename)
        return model_files

    def resolve_url(self, repo: str, filename: str) -> str:
        return f"{self._base_url}/{repo}/resolve/main/{filename}"

    def get_response(self, url: str) -> requests.Response:
        response = self._session.get(url, headers=self.headers)
        self.logger.info(f"Response status was {response.status_code}")
        response.raise_for_status()
        return response


class HFHubTokenizer(HFHubBase):
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger,
    ):
        super().__init__(model_path, logger)

    @staticmethod
    def list_vocab_files(vocab_type: VocabType) -> tuple[str]:
        if vocab_type == VocabType.SPM.value:
            return HF_TOKENIZER_SPM_FILES
        # NOTE: WPM and BPE are equivalent
        return HF_TOKENIZER_BPE_FILES

    @property
    def default_pre_tokenizer(self) -> tuple[str, ...]:
        return GPT_PRE_TOKENIZER_DEFAULT

    def config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "config.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def tokenizer_model(self, model_repo: str) -> SentencePieceProcessor:
        path = self.model_path / model_repo / "tokenizer.model"
        processor = SentencePieceProcessor()
        processor.LoadFromFile(path.read_bytes())
        return processor

    def tokenizer_config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer_config.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def tokenizer_json(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def get_normalizer(self, model_repo: str) -> None | dict[str, object]:
        normalizer = self.tokenizer_json(model_repo).get("normalizer", dict())
        if normalizer:
            self.logger.info(f"JSON:Normalizer: {json.dumps(normalizer, indent=2)}")
        else:
            self.logger.warn(f"WARN:Normalizer: {normalizer}")
        return normalizer

    def get_pre_tokenizer(self, model_repo: str) -> None | dict[str, object]:
        pre_tokenizer = self.tokenizer_json(model_repo).get("pre_tokenizer")
        if pre_tokenizer:
            self.logger.info(
                f"JSON:PreTokenizer: {json.dumps(pre_tokenizer, indent=2)}"
            )
            return pre_tokenizer
        else:
            self.logger.warn(f"WARN:PreTokenizer: {pre_tokenizer}")
        return pre_tokenizer

    def get_added_tokens(self, model_repo: str) -> None | list[dict[str, object]]:
        added_tokens = self.tokenizer_json(model_repo).get("added_tokens", list())
        if added_tokens:
            self.logger.info(f"JSON:AddedTokens: {json.dumps(added_tokens, indent=2)}")
        else:
            self.logger.warn(f"WARN:PreTokenizer: {added_tokens}")
        return added_tokens

    def get_pre_tokenizer_json_hash(self, model_repo: str) -> None | str:
        tokenizer = self.tokenizer_json(model_repo)
        tokenizer_path = self.model_path / model_repo / "tokenizer.json"
        if tokenizer.get("pre_tokenizer"):
            sha256sum = sha256(str(tokenizer.get("pre_tokenizer")).encode()).hexdigest()
        else:
            return
        self.logger.info(f"Hashed '{tokenizer_path}' as {sha256sum}")
        return sha256sum

    def get_tokenizer_json_hash(self, model_repo: str) -> str:
        tokenizer = self.tokenizer_json(model_repo)
        tokenizer_path = self.model_path / model_repo / "tokenizer.json"
        sha256sum = sha256(str(tokenizer).encode()).hexdigest()
        self.logger.info(f"Hashed '{tokenizer_path}' as {sha256sum}")
        return sha256sum

    def log_tokenizer_json_info(self, model_repo: str) -> None:
        self.logger.info(f"{model_repo}")
        tokenizer = self.tokenizer_json(model_repo)
        for k, v in tokenizer.items():
            if k not in ["added_tokens", "model"]:
                self.logger.info(f"{k}:{json.dumps(v, indent=2)}")
            if k == "model":
                for x, y in v.items():
                    if x not in ["vocab", "merges"]:
                        self.logger.info(f"{k}:{x}:{json.dumps(y, indent=2)}")


class HFHubModel(HFHubBase):
    def __init__(
        self,
        auth_token: None | str,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger,
    ):
        super().__init__(model_path, logger)

        self._request = HFHubRequest(auth_token, model_path, logger)
        self._tokenizer = HFHubTokenizer(model_path, logger)

    @property
    def request(self) -> HFHubRequest:
        return self._request

    @property
    def tokenizer(self) -> HFHubTokenizer:
        return self._tokenizer

    def _request_single_file(
        self, model_repo: str, file_name: str, file_path: pathlib.Path
    ) -> bool:
        # NOTE: Consider optional `force` parameter if files need to be updated.
        # e.g. The model creator updated the vocabulary to resolve an issue or add a feature.
        if file_path.exists():
            self.logger.info(f"skipped - downloaded {file_path} exists already.")
            return False

        # NOTE: Do not use bare exceptions! They mask issues!
        # Allow the exception to occur or explicitly handle it.
        try:
            self.logger.info(f"Downloading '{file_name}' from {model_repo}")
            resolved_url = self.request.resolve_url(model_repo, file_name)
            response = self.request.get_response(resolved_url)
            self.write_file(response.content, file_path)
            self.logger.info(f"Model file successfully saved to {file_path}")
            return True
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Error while downloading '{file_name}': {str(e)}")
            return False

    def _request_listed_files(self, model_repo: str, remote_files: list[str]) -> None:
        for file_name in remote_files:
            dir_path = self.model_path / model_repo
            os.makedirs(dir_path, exist_ok=True)
            self._request_single_file(model_repo, file_name, dir_path / file_name)

    def download_model_files(
        self, model_repo: str, file_extension: ModelFileExtension
    ) -> None:
        filtered_files = self.request.list_filtered_remote_files(
            model_repo, file_extension
        )
        self._request_listed_files(model_repo, filtered_files)

    def download_all_vocab_files(self, model_repo: str, vocab_type: VocabType) -> None:
        vocab_files = self.tokenizer.list_vocab_files(vocab_type)
        self._request_listed_files(model_repo, vocab_files)

    def download_all_model_files(self, model_repo: str) -> None:
        all_files = self.request.list_remote_files(model_repo)
        self._request_listed_files(model_repo, all_files)
