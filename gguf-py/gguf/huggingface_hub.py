import json
import logging
import os
import pathlib
from hashlib import sha256
from typing import Protocol

import requests
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from .constants import HF_TOKENIZER_SPM_FILES


class HFHubBase(Protocol):
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
        self.logger.debug(f"Wrote {len(content)} bytes to {file_path} successfully")


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
            # headers = {
            #   "Authorization": f"Bearer {auth_token}",
            #   "securityStatus": True,
            #   "blobs": True,
            # }
            self._headers = {"Authorization": f"Bearer {auth_token}"}

        # Persist across requests
        self._session = requests.Session()

        # This is read-only
        self._base_url = "https://huggingface.co"

        # NOTE: Cache repeat calls
        self._model_repo = None
        self._model_files = None

    @property
    def headers(self) -> None | dict[str, str]:
        return self._headers

    @property
    def session(self) -> requests.Session:
        return self._session

    @property
    def base_url(self) -> str:
        return self._base_url

    def resolve_url(self, repo: str, filename: str) -> str:
        return f"{self._base_url}/{repo}/resolve/main/{filename}"

    def get_response(self, url: str) -> requests.Response:
        # TODO: Stream requests and use tqdm to output the progress live
        response = self._session.get(url, headers=self.headers)
        self.logger.debug(f"Response status was {response.status_code}")
        response.raise_for_status()
        return response

    def model_info(self, model_repo: str) -> dict[str, object]:
        url = f"{self._base_url}/api/models/{model_repo}"
        return self.get_response(url).json()

    def list_remote_files(self, model_repo: str) -> list[str]:
        # NOTE: Reset the cache if the repo changed
        if self._model_repo != model_repo:
            self._model_repo = model_repo
            self._model_files = []
            for f in self.model_info(self._model_repo)["siblings"]:
                self._model_files.append(f["rfilename"])
            dump = json.dumps(self._model_files, indent=4)
            self.logger.debug(f"Cached remote files: {dump}")
        # Return the cached file listing
        return self._model_files

    def list_filtered_remote_files(
        self, model_repo: str, file_suffix: str
    ) -> list[str]:
        model_files = []
        self.logger.debug(f"Model Repo:{model_repo}")
        self.logger.debug(f"File Suffix:{file_suffix}")
        # NOTE: Valuable files are typically in the root path
        for filename in self.list_remote_files(model_repo):
            path = pathlib.Path(filename)
            if len(path.parents) > 1:
                continue  # skip nested paths
            self.logger.debug(f"Path Suffix: {path.suffix}")
            if path.suffix == file_suffix:
                self.logger.debug(f"File Name: {filename}")
                model_files.append(filename)
        return model_files

    def list_remote_safetensors(self, model_repo: str) -> list[str]:
        # NOTE: HuggingFace recommends using safetensors to mitigate pickled injections
        return [
            part
            for part in self.list_filtered_remote_files(model_repo, ".safetensors")
            if part.startswith("model")
        ]

    def list_remote_bin(self, model_repo: str) -> list[str]:
        # NOTE: HuggingFace is streamlining PyTorch models with the ".bin" extension
        return [
            part
            for part in self.list_filtered_remote_files(model_repo, ".bin")
            if part.startswith("pytorch_model")
        ]

    def list_remote_weights(self, model_repo: str) -> list[str]:
        model_parts = self.list_remote_safetensors(model_repo)
        if not model_parts:
            model_parts = self.list_remote_bin(model_repo)
        self.logger.debug(f"Remote model parts: {model_parts}")
        return model_parts

    def list_remote_tokenizers(self, model_repo: str) -> list[str]:
        return [
            tok
            for tok in self.list_remote_files(model_repo)
            if tok in HF_TOKENIZER_SPM_FILES
        ]


class HFHubTokenizer(HFHubBase):
    def __init__(
        self, model_path: None | str | pathlib.Path, logger: None | logging.Logger
    ):
        super().__init__(model_path, logger)

    @staticmethod
    def list_vocab_files() -> tuple[str, ...]:
        return HF_TOKENIZER_SPM_FILES

    def model(self, model_repo: str) -> SentencePieceProcessor:
        path = self.model_path / model_repo / "tokenizer.model"
        processor = SentencePieceProcessor()
        processor.LoadFromFile(path.read_bytes())
        return processor

    def config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer_config.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def json(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def get_normalizer(self, model_repo: str) -> None | dict[str, object]:
        normalizer = self.json(model_repo).get("normalizer", dict())
        if normalizer:
            self.logger.info(f"JSON:Normalizer: {json.dumps(normalizer, indent=2)}")
        else:
            self.logger.warn(f"WARN:Normalizer: {normalizer}")
        return normalizer

    def get_pre_tokenizer(self, model_repo: str) -> None | dict[str, object]:
        pre_tokenizer = self.json(model_repo).get("pre_tokenizer")
        if pre_tokenizer:
            self.logger.info(
                f"JSON:PreTokenizer: {json.dumps(pre_tokenizer, indent=2)}"
            )
            return pre_tokenizer
        else:
            self.logger.warn(f"WARN:PreTokenizer: {pre_tokenizer}")
        return pre_tokenizer

    def get_added_tokens(self, model_repo: str) -> None | list[dict[str, object]]:
        added_tokens = self.json(model_repo).get("added_tokens", list())
        if added_tokens:
            self.logger.info(f"JSON:AddedTokens: {json.dumps(added_tokens, indent=2)}")
        else:
            self.logger.warn(f"WARN:PreTokenizer: {added_tokens}")
        return added_tokens

    def get_pre_tokenizer_json_hash(self, model_repo: str) -> None | str:
        tokenizer = self.json(model_repo)
        tokenizer_path = self.model_path / model_repo / "tokenizer.json"
        if tokenizer.get("pre_tokenizer"):
            sha256sum = sha256(str(tokenizer.get("pre_tokenizer")).encode()).hexdigest()
        else:
            return
        self.logger.info(f"Hashed '{tokenizer_path}' as {sha256sum}")
        return sha256sum

    def get_tokenizer_json_hash(self, model_repo: str) -> str:
        tokenizer = self.json(model_repo)
        tokenizer_path = self.model_path / model_repo / "tokenizer.json"
        sha256sum = sha256(str(tokenizer).encode()).hexdigest()
        self.logger.info(f"Hashed '{tokenizer_path}' as {sha256sum}")
        return sha256sum

    def log_tokenizer_json_info(self, model_repo: str) -> None:
        self.logger.info(f"{model_repo}")
        tokenizer = self.json(model_repo)
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
    ) -> None:
        # NOTE: Do not use bare exceptions! They mask issues!
        # Allow the exception to occur or explicitly handle it.
        try:
            resolved_url = self.request.resolve_url(model_repo, file_name)
            response = self.request.get_response(resolved_url)
            self.write_file(response.content, file_path)
        except requests.exceptions.HTTPError as e:
            self.logger.debug(f"Error while downloading '{file_name}': {str(e)}")

    def _request_listed_files(
        self, model_repo: str, remote_files: list[str, ...]
    ) -> None:
        for file_name in tqdm(remote_files, total=len(remote_files)):
            dir_path = self.model_path / model_repo
            os.makedirs(dir_path, exist_ok=True)

            # NOTE: Consider optional `force` parameter if files need to be updated.
            # e.g. The model creator updated the vocabulary to resolve an issue or add a feature.
            file_path = dir_path / file_name
            if file_path.exists():
                self.logger.debug(f"skipped - downloaded {file_path} exists already.")
                continue  # skip existing files

            self.logger.debug(f"Downloading '{file_name}' from {model_repo}")
            self._request_single_file(model_repo, file_name, file_path)
            self.logger.debug(f"Model file successfully saved to {file_path}")

    def config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "config.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def architecture(self, model_repo: str) -> str:
        # NOTE: Allow IndexError to be raised because something unexpected happened.
        # The general assumption is there is only a single architecture, but
        # merged models may have multiple architecture types. This means this method
        # call is not guaranteed.
        try:
            return self.config(model_repo).get("architectures", [])[0]
        except IndexError:
            self.logger.debug(f"Failed to get {model_repo} architecture")
            return str()

    def download_model_weights(self, model_repo: str) -> None:
        remote_files = self.request.list_remote_weights(model_repo)
        self._request_listed_files(model_repo, remote_files)

    def download_model_tokenizers(self, model_repo: str) -> None:
        remote_files = self.request.list_remote_tokenizers(model_repo)
        self._request_listed_files(model_repo, remote_files)

    def download_model_weights_and_tokenizers(self, model_repo: str) -> None:
        # attempt by priority
        self.download_model_weights(model_repo)
        self.download_model_tokenizers(model_repo)

    def download_all_repository_files(self, model_repo: str) -> None:
        all_files = self.request.list_remote_files(model_repo)
        self._request_listed_files(model_repo, all_files)
