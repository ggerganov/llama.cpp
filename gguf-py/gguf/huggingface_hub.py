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


class HFHub:
    def __init__(
        self,
        model_path: None | str | pathlib.Path,
        auth_token: str,
        logger: None | logging.Logger
    ):
        # Set the model path
        if model_path is None:
            model_path = "models"
        self._model_path = model_path

        # Set the logger
        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Set the hub api
        self._hub = HFHubRequest(auth_token, logger)

        # NOTE: Required for getting model_info
        login(auth_token, add_to_git_credential=True)

    @staticmethod
    def list_remote_files(model_repo: str) -> list[str]:
        # NOTE: Request repository metadata to extract remote filenames
        return [x.rfilename for x in model_info(model_repo).siblings]

    def list_remote_model_files(self, model_repo: str, model_type: ModelFileType) -> list[str]:
        model_files = []
        self.logger.info(f"Repo:{model_repo}")
        self.logger.debug(f"Match:{MODEL_FILE_TYPE_NAMES[model_type]}:{model_type}")
        for filename in HFModel.list_remote_files(model_repo):
            suffix = pathlib.Path(filename).suffix
            self.logger.debug(f"Suffix: {suffix}")
            if suffix == MODEL_FILE_TYPE_NAMES[model_type]:
                self.logger.info(f"File: {filename}")
                model_files.append(filename)
        return model_files

    @property
    def hub(self) -> HFHubRequest:
        return self._hub

    @property
    def model_path(self) -> pathlib.Path:
        return pathlib.Path(self._model_path)

    @model_path.setter
    def model_path(self, value: pathlib.Path):
        self._model_path = value


class HFTokenizer(HFHub):
    def __init__(self, model_path: str, auth_token: str, logger: logging.Logger):
        super().__init__(model_path, auth_token, logger)

    @staticmethod
    def get_vocab_filenames(vocab_type: VocabType) -> tuple[str]:
        if vocab_type == VocabType.SPM:
            return HF_TOKENIZER_SPM_FILES
        # NOTE: WPM and BPE are equivalent
        return HF_TOKENIZER_BPE_FILES

    @staticmethod
    def get_vocab_name(vocab_type: VocabType) -> str:
        return VOCAB_TYPE_NAMES.get(vocab_type)

    @staticmethod
    def get_vocab_enum(vocab_name: str) -> VocabType:
        return {
            "SPM": VocabType.SPM,
            "BPE": VocabType.BPE,
            "WPM": VocabType.WPM,
        }.get(vocab_name, VocabType.NON)

    def config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "config.json"
        return json.loads(path.read_text(encoding='utf-8'))

    def tokenizer_model(self, model_repo: str) -> SentencePieceProcessor:
        path = self.model_path / model_repo / "tokenizer.model"
        processor = SentencePieceProcessor()
        processor.LoadFromFile(path.read_bytes())
        return processor

    def tokenizer_config(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer_config.json"
        return json.loads(path.read_text(encoding='utf-8'))

    def tokenizer_json(self, model_repo: str) -> dict[str, object]:
        path = self.model_path / model_repo / "tokenizer.json"
        return json.loads(path.read_text(encoding='utf-8'))

    def get_normalizer(self, model_repo: str) -> None | dict[str, object]:
        normalizer = self.tokenizer_json(model_repo).get("normalizer", dict())
        if normalizer:
            self.logger.info(f"JSON:Normalizer: {json.dumps(normalizer, indent=2)}")
        else:
            self.logger.warn(f"WARN:Normalizer: {normalizer}")
        return normalizer

    def get_pre_tokenizer(self, model_repo: str) -> None | dict[str, object]:
        pre_tokenizer = self.tokenizer_json(model_repo).get("pre_tokenizer", dict())
        if pre_tokenizer:
            self.logger.info(f"JSON:PreTokenizer: {json.dumps(pre_tokenizer, indent=2)}")
        else:
            self.logger.warn(f"WARN:PreTokenizer: {pre_tokenizer}")
        return pre_tokenizer

    def get_added_tokens(self, model_repo: str) -> None | list[dict[str, object]]:
        added_tokens = self.tokenizer_json(model_repo).get("pre_tokenizer", list())
        if added_tokens:
            self.logger.info(f"JSON:AddedTokens: {json.dumps(added_tokens, indent=2)}")
        else:
            self.logger.warn(f"WARN:PreTokenizer: {added_tokens}")
        return added_tokens

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


class HFModel(HFHub):
    def __init__(
        self,
        auth_token: str,
        model_path: None | str | pathlib.Path,
        logger: None | logging.Logger
    ):
        super().__init__(model_path, auth_token, logger)
        self._tokenizer = HFTokenizer(model_path, auth_token, logger)

    @property
    def model_type(self) -> ModelFileType:
        return ModelFileType

    @property
    def tokenizer(self) -> HFTokenizer:
        return self._tokenizer

    def get_vocab_file(
        self, model_repo: str, file_name: str, file_path: pathlib.Path,
    ) -> bool:
        # NOTE: Do not use bare exceptions! They mask issues!
        # Allow the exception to occur or explicitly handle it.
        resolve_url = self.hub.resolve_url(model_repo, file_name)
        response = self.hub.download_file(resolve_url)
        self.hub.write_file(response.content, file_path)
        self.logger.info(f"Downloaded tokenizer {file_name} from {model_repo}")

    def get_all_vocab_files(self, model_repo: str, vocab_type: VocabType) -> None:
        vocab_list = self.tokenizer.get_vocab_filenames(vocab_type)
        for vocab_file in vocab_list:
            dir_path = self.model_path / model_repo
            file_path = dir_path / vocab_file
            os.makedirs(dir_path, exist_ok=True)
            self.get_vocab_file(model_repo, vocab_file, file_path)

    def get_model_file(self, model_repo: str, file_name: str, file_path: pathlib.Path) -> bool:
        pass

    def get_all_model_files(self) -> None:
        pass
