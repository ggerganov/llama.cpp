import json
import os
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file


@pytest.mark.parametrize(("dtype", "outtype", "expected_type"), [
    (torch.float16, "auto", gguf.GGMLQuantizationType.F16),
    (torch.bfloat16, "auto", gguf.GGMLQuantizationType.BF16),
    (torch.float32, "auto", gguf.GGMLQuantizationType.F16),
    (torch.float32, "f32", gguf.GGMLQuantizationType.F32),
])
def test_lora_conversion_outtype(tmp_path: Path, dtype: torch.dtype, outtype: str, expected_type: gguf.GGMLQuantizationType):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 64,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-5,
    }), encoding="utf-8")

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": ["up_proj"],
        "r": 4,
        "lora_alpha": 8,
    }), encoding="utf-8")
    lora_a = (torch.arange(128).reshape(4, 32) / 16).to(dtype)
    lora_b = (torch.arange(256).reshape(64, 4) / 32).to(dtype)
    save_file({
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": lora_a,
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": lora_b,
    }, adapter / "adapter_model.safetensors")

    script = Path(__file__).resolve().parents[2] / "convert_lora_to_gguf.py"
    output = tmp_path / "adapter.gguf"
    result = subprocess.run([
        sys.executable, str(script), str(adapter),
        "--base", str(base), "--outtype", outtype, "--outfile", str(output),
    ], capture_output=True, text=True, env={
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    assert result.returncode == 0, result.stdout + result.stderr

    expected = {
        "blk.0.ffn_up.weight.lora_a": lora_a.float().numpy(),
        "blk.0.ffn_up.weight.lora_b": lora_b.float().numpy(),
    }
    reader = gguf.GGUFReader(output)
    assert {tensor.name for tensor in reader.tensors} == expected.keys()
    for tensor in reader.tensors:
        assert tensor.tensor_type == expected_type
        np.testing.assert_array_equal(gguf.dequantize(tensor.data, tensor.tensor_type), expected[tensor.name])
