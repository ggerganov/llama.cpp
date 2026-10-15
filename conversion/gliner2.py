from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, SentencePieceTokenTypes, TextModel, gguf, jinja_str_or_json


def _is_gliner2_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "encoder_config" / "config.json").is_file() or not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("model_type") in ("extractor", "gliner2") and config.get("architecture") == "span"


@ModelBase.register_hparams_loader(_is_gliner2_checkpoint)
def _load_gliner2_hparams(dir_model: Path) -> dict[str, Any]:
    with open(dir_model / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    hparams = ModelBase.load_hparams(dir_model / "encoder_config", False, guess=False)
    if config.get("token_pooling") != "first" or config.get("use_moe") or config.get("max_len") is not None:
        raise ValueError("Unsupported GLiNER processor configuration")
    hparams["architectures"] = ["GLiNERDecideModel"]
    return hparams


@ModelBase.register("GLiNERDecideModel")
@ModelBase.example("fastino/GLiNER2.5-Decide")
class GLiNERDecideModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GLINER2

    def set_gguf_parameters(self):
        h = self.hparams
        if (h.get("model_type") != "deberta-v2" or not h.get("relative_attention")
                or not h.get("share_att_key") or set(h.get("pos_att_type", [])) != {"c2p", "p2c"}
                or h.get("position_biased_input") or h.get("type_vocab_size", 0) or h.get("conv_kernel_size", 0)
                or h.get("norm_rel_ebd") != "layer_norm" or h.get("hidden_act") != "gelu"
                or h.get("embedding_size", h["hidden_size"]) != h["hidden_size"]):
            raise ValueError("Unsupported DeBERTa-v2 encoder configuration")
        super().set_gguf_parameters()
        self.gguf_writer.add_causal_attention(False)
        self.gguf_writer.add_pooling_type(gguf.PoolingType.NONE)
        self.gguf_writer.add_relative_attn_buckets_count(h["position_buckets"])
        distance = h["max_relative_positions"]
        self.gguf_writer.add_relative_attn_max_distance(distance if distance > 0 else h["max_position_embeddings"])
        self.gguf_writer.add_decision_type(gguf.DecisionType.GLINER)

    def set_vocab(self):
        with open(self.dir_model / "tokenizer.json", encoding="utf-8") as f:
            tokenizer = json.load(f)
        model = tokenizer["model"]
        if model["type"] != "Unigram" or model.get("byte_fallback"):
            raise ValueError("GLiNER needs a Unigram tokenizer without byte fallback")
        expected_normalizer = {"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"Regex": r"\s{2,}|[\n\r\t]"}, "content": " "},
            {"type": "NFC"}, {"type": "Strip", "strip_left": False, "strip_right": True},
        ]}
        # Some tokenizer files escape the regular expression twice.
        normalizer = tokenizer["normalizer"]
        pattern = normalizer["normalizers"][0]["pattern"]["Regex"]
        if pattern == r"\\s{2,}|[\\n\\r\\t]":
            normalizer["normalizers"][0]["pattern"]["Regex"] = r"\s{2,}|[\n\r\t]"
        if normalizer != expected_normalizer or tokenizer["pre_tokenizer"] != {
            "type": "Sequence", "pretokenizers": [{"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always", "split": True}],
        }:
            raise ValueError("Unsupported GLiNER tokenizer pipeline")
        tokens = [entry[0] for entry in model["vocab"]]
        scores = [entry[1] for entry in model["vocab"]]
        types = [SentencePieceTokenTypes.NORMAL] * len(tokens)
        types[model["unk_id"]] = SentencePieceTokenTypes.UNKNOWN
        for entry in tokenizer["added_tokens"]:
            idx = entry["id"]
            if any(entry.get(key) for key in ("single_word", "lstrip", "rstrip")) or (entry.get("normalized") and idx != model["unk_id"]) or not entry["special"]:
                raise ValueError("Unsupported GLiNER added token configuration")
            if idx == len(tokens):
                tokens.append(entry["content"])
                scores.append(0.0)
                types.append(SentencePieceTokenTypes.CONTROL)
            elif 0 <= idx < len(tokens) and tokens[idx] == entry["content"]:
                types[idx] = SentencePieceTokenTypes.UNKNOWN if idx == model["unk_id"] else SentencePieceTokenTypes.CONTROL
            else:
                raise ValueError("Invalid GLiNER added token id")
        if len(tokens) != self.hparams["vocab_size"]:
            raise ValueError("GLiNER tokenizer and encoder vocabulary sizes differ")
        self.gguf_writer.add_tokenizer_model("t5")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(types)
        self.gguf_writer.add_unk_token_id(model["unk_id"])
        self.gguf_writer.add_pad_token_id(self.hparams["pad_token_id"])
        self.gguf_writer.add_eos_token_id(tokens.index("[SEP]"))
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)
        self.gguf_writer.add_add_space_prefix(True)
        self.gguf_writer.add_remove_extra_whitespaces(True)
        self.gguf_writer.add_normalizer_nfc(True)
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @staticmethod
    def _systemone_template() -> str:
        instructions = jinja_str_or_json("q.instructions")
        description = jinja_str_or_json("o.description")
        return (
            "{% for q in questions %}{% if not loop.first %}[SEP_STRUCT]{{ sep }}{% endif %}"
            "({{ sep }}[P]{{ sep }}{{ q.id }}{% if q.instructions %}: " + instructions + "{% endif %}"
            "{% for o in q.options %}{% if o.description is not none %} [DESCRIPTION] {{ o.key }}: " + description + "{% endif %}{% endfor %}"
            "{{ sep }}({{ sep }}{% for o in q.options %}{{ mark_option }}[L]{{ sep }}{{ o.key }}{{ sep }}{% endfor %}"
            "){{ sep }}){{ sep }}{% endfor %}[SEP_TEXT]{{ sep }}{% for word in words %}{{ word }}{{ sep }}{% endfor %}"
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("encoder.LayerNorm."):
            return
        if name == "encoder.rel_embeddings.weight":
            # The relative embedding norm is constant across requests.
            data_torch = torch.nn.functional.layer_norm(
                LazyTorchTensor.to_eager(data_torch).float(), (self.hparams["hidden_size"],),
                LazyTorchTensor.to_eager(self.model_tensors["encoder.LayerNorm.weight"]()).float(),
                LazyTorchTensor.to_eager(self.model_tensors["encoder.LayerNorm.bias"]()).float(), self.hparams["layer_norm_eps"],
            )
        yield from super().modify_tensors(data_torch, name, bid)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith(("encoder.", "classifier.")):
            return None
        name = name.removeprefix("encoder.")
        return super().filter_tensors((name, gen))
