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

    def write_file(self, content: bytes, path: pathlib.Path) -> None:
        with open(path, 'wb') as f:
            f.write(content)
        self.logger.info(f"Wrote {len(content)} bytes to {path} successfully")

    def resolve_path(self, repo: str, file: str) -> str:
        return f"{self._base_url}/{repo}/resolve/main/{file}"

    def download_file(self, repo: str, file: str):
        resolve_path = self.resolve_path(repo, file)
        response = self._session.get(resolve_path, headers=self.headers)
        self.logger.info(f"Response status was {response.status_code}")
        response.raise_for_status()
        return response


class HFTokenizerRequest:
    def __init__(
        self,
        dl_path: pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        self._hub = HuggingFaceHub(auth_token, logger)
        self._models = MODEL_REPOS

        if dl_path is None:
            self._download_path = pathlib.Path("models/tokenizers")
        else:
            self._download_path = dl_path

        self._files = ["config.json", "tokenizer_config.json", "tokenizer.json"]

    @property
    def hub(self) -> HuggingFaceHub:
        return self._hub

    @property
    def models(self) -> list[dict[str, object]]:
        return self._models

    @property
    def download_path(self) -> pathlib.Path:
        return self._download_path

    @download_path.setter
    def download_path(self, value: pathlib.Path):
        self._download_path = value

    @property
    def files(self) -> list[str]:
        return self._files

    def download_file_with_auth(self, repo, file, directory):
        response = self.hub.download_file(repo, file)
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        self.hub.write_file(response.content, directory)
