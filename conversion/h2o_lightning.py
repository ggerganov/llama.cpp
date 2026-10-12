from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel
from .qwen3vl import Qwen3VLVisionModel


def _is_h2o_lightning_checkpoint(dir_model: Path) -> bool:
    return (dir_model / "h2o_lightning_shim.py").is_file() and (dir_model / "serve_config.json").is_file()


@ModelBase.register_hparams_loader(_is_h2o_lightning_checkpoint)
def _load_h2o_lightning_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected H2O-Lightning checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["H2OLightningModel"]
    return hparams


@ModelBase.register("H2OLightningModel")
@ModelBase.example("h2oai/h2o-lightning-4b")
class H2OLightningModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    # prompt and calibration follow serve_config.json and h2o_lightning_shim.py of the model repo
    # the labels and the state budget are hard-coded in server-decision.cpp
    _LABELS = (
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        + list("\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03b9\u03ba\u03bb\u03bc\u03bd\u03be\u03bf\u03c0\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9")
        + (
            "in er on re at st en or th le it an ar al the ou is ing es ion ed ic et ro as el ct nd ent id am to "
            "of om im il and ur se ex ad ch ut if em ol ig iv ce od ate ag ay ot us un ul ue ow ew for ab ort um "
            "ame pe tr ck ist he lo be ers con ap ub ass int ly urn av ir nt end de ith out our lic res pt wh ver "
            "age ht ext ess os ect ke rom qu iz op up get ile ata ore ri pro ter ain com art ack est ine ill ind "
            "ere elf orm ult str you pl new ld oc ant ase ave ang are ial act ive ode ost not og ord all ff ont "
            "ime pr ies ure by ire sh ice ast ok cl ard was ip The ide ace ib ac rou we ak ve ath oid ust que ans "
            "pp one ail ach div ell"
        ).split()
    )
    _MAX_STATE_TOKENS = 32000
    _DEFAULT_INSTRUCTIONS = "Answer the question below."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.dir_model / "serve_config.json", "r", encoding="utf-8") as f:
            self._serve_config = json.load(f)
        cfg, prompt = self._serve_config, self._serve_config["prompt"]
        expected = {
            "labels":           (cfg["labels"][:cfg.get("max_options", len(cfg["labels"]))], self._LABELS),
            "state_compact":    (prompt.get("state_compact", True), True),
            "max_state_tokens": (prompt.get("max_state_tokens"), self._MAX_STATE_TOKENS),
            "layout":           (prompt.get("layout", "plain"), "plain"),
            "chat_template":    (prompt.get("chat_template", True), True),
            "enable_thinking":  (prompt.get("enable_thinking", False), False),
        }
        for name, (got, want) in expected.items():
            if isinstance(want, list) and got != want:
                i = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), min(len(got), len(want)))
                raise ValueError(f"serve_config.json: {name} differ from the h2o-lightning decision type at index {i}")
            if got != want:
                raise ValueError(f"serve_config.json: {name} is {got!r}, the h2o-lightning decision type expects {want!r}")

    def get_vocab_base_pre(self, tokenizer) -> str:
        # transformers loads this checkpoint as Qwen2Tokenizer: the model was trained and is served with the Qwen2 split
        pre = super().get_vocab_base_pre(tokenizer)
        if pre != "qwen35":
            raise ValueError(f"expected the Qwen3.5 tokenizer, got pre-tokenizer {pre!r}")
        return "qwen2"

    def set_vocab(self):
        super().set_vocab()
        self.gguf_writer.add_chat_template([{"name": "systemone", "template": self._systemone_template()}])

    def _systemone_template(self) -> str:
        # the server gives the template texts rendered as the shim renders them, it only lays them out
        prompt = self._serve_config["prompt"]
        nl = "{{ '\\n' }}"
        description = (
            "{% if type == 'noul' %}"
            "{% if o.description %}{{ o.description }}{% elif o.key == 'true' %}the statement holds{% else %}it does not{% endif %}"
            "{% elif o.description is none %}{{ o.key }}{% else %}{{ o.description }}{% endif %}"
        )
        user = (
            "{% set user %}record: {{ state }}" + nl
            + "question: {{ instructions if instructions else '" + self._DEFAULT_INSTRUCTIONS + "' }}" + nl
            + "options:{% for o in options %}" + nl + "{{ o.label }}) {{ o.key }}: " + description + "{% endfor %}{% endset %}"
        )
        return (
            "<|im_start|>system" + nl + prompt["system"].strip() + "<|im_end|>" + nl
            + "<|im_start|>user" + nl + "{% for image in images %}{{ image }}{% endfor %}" + user + "{{ user | trim }}<|im_end|>" + nl
            + "<|im_start|>assistant" + nl + "<think>{{ '\\n\\n' }}</think>{{ '\\n\\n' }}" + prompt["prefill"]
        )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_decision_type(gguf.DecisionType.H2O_LIGHTNING)
        for name in ("choice", "noul", "score"):
            self.gguf_writer.add_decision_temperature(name, self._serve_config["temperature"][name])
        # a request with images has this temperature for all question types
        self.gguf_writer.add_decision_temperature("image", self._serve_config["image"]["temperature"])


@ModelBase.register("H2OLightningModel")
class H2OLightningVisionModel(Qwen3VLVisionModel):
    pass
