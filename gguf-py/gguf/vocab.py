from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .gguf_writer import GGUFWriter

class SpecialVocab:
    load_merges: bool = False
    merges: list[str] = []
    special_token_types: tuple[str, ...] = ('bos', 'eos', 'unk', 'sep', 'pad')
    special_token_ids: dict[str, int] = {}
    n_vocab: int | None = None

    def __init__(
        self, path: str | os.PathLike[str], load_merges: bool = False,
        special_token_types: tuple[str, ...] | None = None,
        n_vocab: int | None = None,
    ):
        self.special_token_ids = {}
        self.n_vocab = n_vocab
        self.load_merges = load_merges
        if special_token_types is not None:
            self.special_token_types = special_token_types
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not self._try_load_from_tokenizer_json(path):
            self._try_load_from_config_json(path)

    def _set_special_token(self, typ: str, tid: Any) -> None:
        if not isinstance(tid, int) or tid < 0:
            return
        if self.n_vocab is None or tid < self.n_vocab:
            self.special_token_ids[typ] = tid
            return
        print(f'gguf: WARNING: Special token type {typ}, id {tid} out of range, must be under {self.n_vocab} - skipping',
            file = sys.stderr)


    def _try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.json'
        if not tokenizer_file.is_file():
            return False
        with open(tokenizer_file, encoding = 'utf-8') as f:
            tokenizer = json.load(f)
        if self.load_merges:
            merges = tokenizer.get('model', {}).get('merges')
            if isinstance(merges, list) and len(merges) > 0 and isinstance(merges[0], str):
                self.merges = merges
        tokenizer_config_file = path / 'tokenizer_config.json'
        added_tokens = tokenizer.get('added_tokens')
        if added_tokens is None or not tokenizer_config_file.is_file():
            return True
        with open(tokenizer_config_file, encoding = 'utf-8') as f:
            tokenizer_config = json.load(f)
        for typ in self.special_token_types:
            entry = tokenizer_config.get(f'{typ}_token')
            if isinstance(entry, str):
                tc_content = entry
            elif isinstance(entry, dict):
                entry_content = entry.get('content')
                if not isinstance(entry_content, str):
                    continue
                tc_content = entry_content
            else:
                continue
            # We only need the first match here.
            maybe_token_id = next((
                atok.get('id') for atok in added_tokens
                if atok.get('content') == tc_content), None)
            self._set_special_token(typ, maybe_token_id)
        return True

    def _try_load_from_config_json(self, path: Path) -> bool:
        config_file = path / 'config.json'
        if not config_file.is_file():
            return False
        with open(config_file, encoding = 'utf-8') as f:
            config = json.load(f)
        for typ in self.special_token_types:
            self._set_special_token(typ, config.get(f'{typ}_token_id'))
        return True

    def add_to_gguf(self, gw: GGUFWriter, quiet: bool = False) -> None:
        if len(self.merges) > 0:
            if not quiet:
                print(f'gguf: Adding {len(self.merges)} merge(s).')
            gw.add_token_merges(self.merges)
        for typ, tokid in self.special_token_ids.items():
            handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if handler is None:
                print(f'gguf: WARNING: No handler for special token type {typ} with id {tokid} - skipping', file = sys.stderr)
                continue
            if not quiet:
                print(f'gguf: Setting special token type {typ} to {tokid}')
            handler(tokid)

    def __repr__(self) -> str:
        return f'<SpecialVocab with {len(self.merges)} merges and special tokens {self.special_token_ids or "unset"}>'
