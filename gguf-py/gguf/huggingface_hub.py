import logging
import os
import pathlib

import requests

from .constants import MODEL_REPOS, TokenizerType


class HuggingFaceHub:
    def __init__(self, auth_token: None | str, logger: None | logging.Logger):
        # Set headers if authentication is available
        if auth_token is None:
            self._headers = None
        else:
            self._headers = {"Authorization": f"Bearer {auth_token}"}

        # Set the logger
        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger("huggingface-hub")
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


class HFTokenizerRequest:
    def __init__(
        self,
        dl_path: None | str | pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        self._hub = HuggingFaceHub(auth_token, logger)

        if dl_path is None:
            self._local_path = pathlib.Path("models/tokenizers")
        else:
            self._local_path = dl_path

        # Set the logger
        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger("hf-tok-req")
        self.logger = logger

    @property
    def hub(self) -> HuggingFaceHub:
        return self._hub

    @property
    def models(self) -> list[dict[str, object]]:
        return MODEL_REPOS

    @property
    def tokenizer_type(self) -> TokenizerType:
        return TokenizerType

    @property
    def local_path(self) -> pathlib.Path:
        return self._local_path

    @local_path.setter
    def local_path(self, value: pathlib.Path):
        self._local_path = value

    def resolve_filenames(self, tokt: TokenizerType) -> tuple[str]:
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
            self.logger.error(f"Failed to download tokenizer {model['name']}: {e}")

    def download_model(self) -> None:
        for model in self.models:
            os.makedirs(f"{self.local_path}/{model['name']}", exist_ok=True)
            filenames = self.resolve_filenames(model['tokt'])
            for filename in filenames:
                filepath = pathlib.Path(f"{self.local_path}/{model['name']}/{filename}")
                if filepath.is_file():
                    self.logger.info(f"skipped pre-existing tokenizer {model['name']} at {filepath}")
                    continue
                self.resolve_tokenizer_model(filename, filepath, model)
