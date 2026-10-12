from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, gguf, logger
from .qwen import Qwen3Model, Qwen3_5TextModel


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_vllm_decision2_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    return _read_json(dir_model / "config.json").get("decision_format") == "vllm-sr-decision"


def _backbone_dir(dir_model: Path, package: dict[str, Any]) -> Path:
    backbone = package["backbone"]
    if "config" in backbone:
        return (dir_model / backbone["config"]).parent

    # the backbone is not in the package, it is pinned by repository and revision
    base = _read_json(dir_model / package["manifest"])["base"]
    if base["repo_id"] != backbone["repository"] or base["revision"] != backbone["revision"]:
        raise ValueError("the backbone of the package does not match its manifest")
    from huggingface_hub import snapshot_download
    logger.info(f"gguf: fetching the backbone {backbone['repository']} at {backbone['revision']}")
    return Path(snapshot_download(backbone["repository"], revision=backbone["revision"],
                                  allow_patterns=["config.json", "model*.safetensors", "model.safetensors.index.json"]))


@ModelBase.register_hparams_loader(_is_vllm_decision2_checkpoint)
def _load_vllm_decision2_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected vllm-sr Decision 2.0 checkpoint")
    package = _read_json(dir_model / "config.json")
    hparams = ModelBase.load_hparams(_backbone_dir(dir_model, package), False, guess=False)
    text_config = hparams.get("text_config", hparams)
    hparams["architectures"] = ["VllmDecision2Model" if text_config["model_type"] == "qwen3" else "VllmDec2Qwen35Model"]
    # the model refuses longer inputs
    for config in {id(hparams): hparams, id(text_config): text_config}.values():
        config["max_position_embeddings"] = min(config["max_position_embeddings"], package["max_input_tokens"])
    hparams["decision"] = _read_json(dir_model / package["model_config"])
    hparams["decision"]["score_bias"] = _read_json(dir_model / "score_bias.json")["offsets"] if (dir_model / "score_bias.json").is_file() else {}
    return hparams


class _VllmDecision2Mixin(ModelBase if TYPE_CHECKING else object):
    # torch.nn.LayerNorm default, used by the head
    _HEAD_NORM_EPS = 1e-5
    _HEAD_ARCHITECTURE = "-text-endpoints-global-query-shared-bilinear-mlp"

    def __init__(self, dir_model: Path, *args, **kwargs):
        package = _read_json(dir_model / "config.json")
        self._dir_backbone = _backbone_dir(dir_model, package)
        # "layers.N.<module>.weight" -> (A, B), merged into the backbone weights
        self._lora: dict[str, tuple[Tensor, Tensor]] = {}
        self._lora_scale = 1.0
        if "adapter" in package:
            self._load_adapter(dir_model, package["adapter"])
        super().__init__(dir_model, *args, **kwargs)
        head = self.hparams["decision"]
        if not head["architecture"].endswith(self._HEAD_ARCHITECTURE) or head.get("head_variant", "shared") != "shared":
            raise ValueError(f"unsupported decision head: {head['architecture']}")

    def _load_adapter(self, dir_model: Path, adapter: dict[str, Any]):
        from safetensors.torch import load_file
        config = _read_json(dir_model / adapter["config"])
        if config["peft_type"] != "LORA" or config.get("bias", "none") != "none" or config.get("use_dora") or config.get("use_rslora") \
                or config.get("rank_pattern") or config.get("alpha_pattern") or config.get("modules_to_save"):
            raise ValueError("unsupported LoRA adapter")
        self._lora_scale = config["lora_alpha"] / config["r"]
        halves: dict[str, dict[str, Tensor]] = {}
        for path in adapter["weights"]:
            for name, data in load_file(dir_model / path).items():
                # base_model.model.layers.N.<module>.lora_A.weight
                base, _, part = name[name.index("layers."):].partition(".lora_")
                halves.setdefault(base + ".weight", {})[part[0]] = data
        for key, half in halves.items():
            if set(half) != {"A", "B"}:
                raise ValueError(f"incomplete LoRA adapter: {key}")
            self._lora[key] = (half["A"], half["B"])
        logger.info(f"gguf: merging the LoRA adapter of {len(self._lora)} tensors, scale {self._lora_scale}")

    def index_tensors(self, *args, **kwargs) -> dict[str, Callable[[], Tensor]]:
        # the backbone weights are in their own directory
        dir_model, self.dir_model = self.dir_model, self._dir_backbone
        try:
            return super().index_tensors(*args, **kwargs)
        finally:
            self.dir_model = dir_model

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        yield from super().get_tensors()
        from safetensors.torch import load_file
        for name, data in load_file(self.dir_model / "decision_head.safetensors").items():
            yield "decision_head." + name, data

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the head has plain LayerNorms, the backbone may offset its norm weights
        if name.startswith("decision_head."):
            return [(self.map_tensor_name(name), data_torch)]
        lora = self._lora.pop(name[name.index("layers."):], None) if "layers." in name else None
        if lora is not None:
            a, b = (LazyTorchTensor.from_eager(t.float()) for t in lora)
            if not isinstance(data_torch, LazyTorchTensor):
                data_torch = LazyTorchTensor.from_eager(data_torch)
            data_torch = data_torch.float() + self._lora_scale * (b @ a)
        return super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._lora:
            raise ValueError(f"LoRA tensors without a backbone weight: {sorted(self._lora)[:3]}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the head is computed in F32
        if new_name.startswith("decision."):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    @classmethod
    def _systemone_template(cls) -> str:
        # prompt follows decision2/_vendor/dev2model/decision_model.py (segments) of the model repo
        # the pieces of the prompt are tokenized one by one, the server gives the text that separates them (sep)
        # and the text that starts the span of a question or of an option (mark_question, mark_option)
        # the head reads the last token of each option and the last token of the question
        # the start of an option is a piece of its own, so that the spans of two options are not joined
        # the keys of JSON objects are given in sorted order
        def text(value: str) -> str:
            return "{{ " + json.dumps(value) + " }}"

        def render(name: str) -> str:
            return "{{ " + name + " if " + name + " is string else " + name + " | tojson(separators=[',', ':']) }}"

        option = (
            "{% set d = o.description %}"
            "{% if q.type == 'noul' and d is none %}{% set d = 'Yes' if o.key == 'true' else 'No' %}{% endif %}"
            "{{ {'description': d, 'key': o.key} | tojson(separators=[',', ':']) }}"
        )
        return (
            "{% for q in questions %}"
            + text("Context:\n") + render("state")
            + text("\n\nTask type: ") + "{{ q.type }}" + text("\nQuestion:\n") + render("q.instructions") + text("\nOptions:")
            + "{% for o in q.options %}{{ sep }}" + text("\n<option>\n") + "{{ sep }}{{ mark_option }}" + option + text("\n</option>") + "{% endfor %}"
            + "{{ sep }}{{ mark_question }}" + text("\n\nSelect the single option best supported by the context and instructions.\nDecision:")
            + "{% endfor %}"
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.VLLM_DECISION2)
        self.gguf_writer.add_layer_norm_eps(self._HEAD_NORM_EPS)
        for offsets in self.hparams["decision"]["score_bias"].values():
            self.gguf_writer.add_decision_score_bias(offsets)


@ModelBase.register("VllmDecision2Model")
@ModelBase.example("vllm-sr/Decision-2.0-Kai-0.6B")
class VllmDecision2Model(_VllmDecision2Mixin, Qwen3Model):
    model_arch = gguf.MODEL_ARCH.VLLM_DECISION2


@ModelBase.register("VllmDec2Qwen35Model")
@ModelBase.example("vllm-sr/Decision-2.0-Lux-9B", "vllm-sr/Decision-2.0-Vega-27B")
class VllmDec2Qwen35Model(_VllmDecision2Mixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.VLLM_DEC2_QWEN35
    no_mtp = True  # the checkpoint has no MTP head

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # the head replaces the LM head
        if name == "lm_head.weight":
            return None
        # a packaged backbone is a bare Qwen3_5TextModel, its tensors have no "model." prefix
        if not name.startswith("model."):
            name = "model." + name
        return super().filter_tensors((name, gen))
