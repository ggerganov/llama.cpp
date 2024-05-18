import logging
import pathlib

import requests


class HuggingFaceHub:
    def __init__(self, auth_token: None | str):
        # Set headers if authentication is available
        if auth_token is None:
            self._headers = {}
        else:
            self._headers = {"Authorization": f"Bearer {auth_token}"}

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

    def resolve_path(self, repo: str, file: str) -> str:
        return f"{self._base_url}/{repo}/resolve/main/{file}"

    def download_file(self, repo: str, file: str):
        endpoint = self.resolve_path(repo, file)
        response = self._session.get(endpoint, headers=self.headers)
        response.raise_for_status()
        return response
