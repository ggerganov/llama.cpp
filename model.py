import os
import re
import sys
import json
import gguf
import torch
import contextlib
import numpy as np

from pathlib import Path


class Model:
    def __init__(self, dir_model: Path, ftype: int, fname_out: Path):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_safetensors = self._is_model_safetensors()
        self.num_parts = Model.count_model_parts(self.dir_model, ".safetensors" if self.is_safetensors else ".bin")
        self.part_names = self._get_part_names()
        self.hparams = Model.load_hparams(self.dir_model)
        self.model_arch = self._get_model_architecture()
        self.gguf_writer = gguf.GGUFWriter(fname_out, gguf.MODEL_ARCH_NAMES[self.model_arch])

    def _is_model_safetensors(self) -> bool:
        return Model.count_model_parts(self.dir_model, ".safetensors") > 0

    def _get_part_names(self):
        if self.is_safetensors:
            if self.num_parts == 1:  # there's only one .safetensors file
                return ("model.safetensors",)
            return (f"model-{n:05}-of-{self.num_parts:05}.safetensors" for n in range(1, self.num_parts + 1))
        else:
            if self.num_parts == 1:  # there's only one .bin file
                return ("pytorch_model.bin",)
            return (f"pytorch_model-{n:05}-of-{self.num_parts:05}.bin" for n in range(1, self.num_parts + 1))

    def _get_model_architecture(self):
        arch = self.hparams["architectures"][0]
        if arch == "GPTNeoXForCausalLM":
            return gguf.MODEL_ARCH.GPTNEOX
        if arch == "BloomForCausalLM":
            return gguf.MODEL_ARCH.BLOOM
        if arch == "MPTForCausalLM":
            return gguf.MODEL_ARCH.MPT
        raise NotImplementedError(f'Architecture "{arch}" not supported!')

    def set_vocab(self):
        dir_model = self.dir_model
        hparams = self.hparams
        tokens: list[bytearray] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(dir_model)
        vocab_size = hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        reverse_vocab = {id: encoded_tok for encoded_tok, id in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.USER_DEFINED)
            elif reverse_vocab[i] in added_vocab:
                tokens.append(reverse_vocab[i])
                if tokenizer.added_tokens_decoder[i].special:
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                tokens.append(reverse_vocab[i])
                toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(dir_model, load_merges = True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def get_tensors(self):
        for part_name in self.part_names:
            print("gguf: loading model part '" + part_name + "'")
            if self.is_safetensors:
                from safetensors import safe_open
                ctx = safe_open(self.dir_model / part_name, framework="pt", device="cpu")
            else:
                ctx = contextlib.nullcontext(torch.load(self.dir_model / part_name, map_location="cpu"))

            with ctx as model_part:
                for name in model_part.keys():
                    data = model_part.get_tensor(name) if self.is_safetensors else model_part[name]
                    yield name, data

    def set_gguf_parameters(self):
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_block_count(self.hparams.get("n_layers", self.hparams.get("num_hidden_layers")))
        if "max_position_embeddings" in self.hparams:
            self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        if "hidden_size" in self.hparams:
            self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        if "intermediate_size" in self.hparams:
            self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        if "num_attention_head" in self.hparams:
            self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(self.hparams["use_parallel_residual"] if "use_parallel_residual" in self.hparams else True)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data in self.get_tensors():
            # we don't need these
            if name.endswith(".attention.masked_bias") or name.endswith(".attention.bias") or name.endswith(".attention.rotary_emb.inv_freq"):
                continue

            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            data = data.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
            if new_name is None:
                print("Can not map tensor '" + name + "'")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

            self.gguf_writer.add_tensor(new_name, data)

    @staticmethod
    def count_model_parts(dir_model: Path, prefix: str) -> int:
        num_parts = 0
        for filename in os.listdir(dir_model):
            if filename.endswith(prefix):
                num_parts += 1

        return num_parts

    @staticmethod
    def load_hparams(dir_model):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            hparams = json.load(f)
            return hparams


    @staticmethod
    def from_model_architecture(model_architecture):
        if model_architecture == "StableLMEpochForCausalLM":
            return StableLMModel
        if model_architecture == "GPTNeoXForCausalLM":
            return GPTNeoXModel
        if model_architecture == "BloomForCausalLM":
            return BloomModel
        if model_architecture == "MPTForCausalLM":
            return MPTModel
        return Model

class StableLMModel(Model):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_count(int(self.hparams["rope_pct"]*(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])))
        self.gguf_writer.add_layer_norm_eps(1e-5)


class GPTNeoXModel(Model):
    pass

class BloomModel(Model):
    def set_gguf_parameters(self):
        self.gguf_writer.add_name("Bloom")
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_writer.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_writer.add_embedding_length(n_embed)
        self.gguf_writer.add_feed_forward_length(4 * n_embed)
        self.gguf_writer.add_block_count(self.hparams["n_layer"])
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams["n_layer"]
        tensors = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        has_lm_head = True
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        for name, data in tensors.items():
            if "lm_head.weight" not in tensors.keys() and "output.weight" not in tensors.keys():
                has_lm_head = False

            name = re.sub(r'transformer\.', '', name)

            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            data = data.squeeze().numpy()

            if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
                # Map bloom-style qkv_linear to gpt-style qkv_linear
                # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
                # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
                qkv_weights = data.reshape((n_head, 3, n_embed // n_head, n_embed))
                data = np.concatenate(
                    (qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                     qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                     qkv_weights[:, 2, :, :].reshape((-1, n_embed))),
                    axis=0
                )
                print("re-format attention.linear_qkv.weight")
            elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
                qkv_bias = data.reshape((n_head, 3, n_embed // n_head))
                data = np.concatenate(
                    (qkv_bias[:, 0, :].reshape((n_embed,)),
                     qkv_bias[:, 1, :].reshape((n_embed,)),
                     qkv_bias[:, 2, :].reshape((n_embed,))),
                    axis=0
                )
                print("re-format attention.linear_qkv.bias")

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print("Can not map tensor '" + name + "'")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(name, "=>", new_name + ", shape = " + str(data.shape) + ", " + str(old_dtype) + " --> " + str(data.dtype))

            self.gguf_writer.add_tensor(new_name, data)

            if not has_lm_head and name == "word_embeddings.weight":
                self.gguf_writer.add_tensor("output.weight", data)
                print(name, "=>", "output.weight" + ", shape = " + str(data.shape) + ", " + str(old_dtype) + " --> " + str(data.dtype))  # noqa

class MPTModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_seq_len"])
        self.gguf_writer.add_embedding_length(self.hparams["d_model"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_writer.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_writer.add_head_count_kv(kv_n_heads)
        self.gguf_writer.add_layer_norm_eps(1e-05)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_writer.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        self.gguf_writer.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])
