import os
import re
import sys
import json
import gguf
import torch
import contextlib
import numpy as np

from pathlib import Path
from typing import TypeAlias, Any

NDArray: TypeAlias = 'np.ndarray[Any, Any]'


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
        if arch in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return gguf.MODEL_ARCH.BAICHUAN
        if arch == "FalconForCausalLM":
            return gguf.MODEL_ARCH.FALCON
        if arch == "GPTBigCodeForCausalLM":
            return gguf.MODEL_ARCH.STARCODER
        if arch == "GPTRefactForCausalLM":
            return gguf.MODEL_ARCH.REFACT
        if arch == "PersimmonForCausalLM":
            return gguf.MODEL_ARCH.PERSIMMON

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

        special_vocab = gguf.SpecialVocab(dir_model, load_merges=True)
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
        self.gguf_writer.add_block_count(self.hparams.get(
            "n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer"))))
        if "max_position_embeddings" in self.hparams:
            self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        if "hidden_size" in self.hparams:
            self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        if "intermediate_size" in self.hparams:
            self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        if "num_attention_head" in self.hparams:
            self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(
            self.hparams["use_parallel_residual"] if "use_parallel_residual" in self.hparams else True)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
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
        if model_architecture in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return BaichuanModel
        if model_architecture == "FalconForCausalLM":
            return FalconModel
        if model_architecture == "GPTBigCodeForCausalLM":
            return StarCoderModel
        if model_architecture == "GPTRefactForCausalLM":
            return RefactModel
        if model_architecture == "PersimmonForCausalLM":
            return PersimmonModel
        return Model


class StableLMModel(Model):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rope_pct"]*(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])))
        self.gguf_writer.add_layer_norm_eps(1e-5)


class GPTNeoXModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(
            int(self.hparams["rotary_pct"]*(self.hparams["hidden_size"]//self.hparams["num_attention_heads"])))
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_parallel_residual(
            self.hparams["use_parallel_residual"] if "use_parallel_residual" in self.hparams else True)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_eps"])


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

            print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

            self.gguf_writer.add_tensor(new_name, data)

            # note: MPT output is tied to (same as) wte in original model;
            # for easier implementation in llama.cpp it's duplicated in GGUF, though :/
            if new_name == "token_embd.weight":
                self.gguf_writer.add_tensor("output.weight", data)


class BaichuanModel(Model):
    def set_vocab(self):
        from sentencepiece import SentencePieceProcessor  # type: ignore[import]
        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        tokenizer_model_file = self.dir_model / 'tokenizer.model'
        if not tokenizer_model_file.is_file():
            print(f'Error: Missing {tokenizer_model_file}', file=sys.stderr)
            sys.exit(1)

        # vocab type sentencepiece
        print("gguf: get sentencepiece tokenizer vocab, scores and token types")

        tokenizer = SentencePieceProcessor(str(tokenizer_model_file))
        vocab_size = self.hparams.get('vocab_size')
        if vocab_size is None:
            vocab_size = tokenizer.vocab_size()

        for i in range(vocab_size):
            text: bytes
            score: float

            piece = tokenizer.id_to_piece(i)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(i)

            toktype = 1  # defualt to normal token type
            if tokenizer.is_unknown(i):
                toktype = 2
            if tokenizer.is_control(i):
                toktype = 3

            # toktype = 4 is user-defined = tokens from added_tokens.json

            if tokenizer.is_unused(i):
                toktype = 5
            if tokenizer.is_byte(i):
                toktype = 6

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        added_tokens_file = self.dir_model / 'added_tokens.json'
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                addtokens_json = json.load(f)

                print("gguf: get added tokens")

                for key in addtokens_json:
                    tokens.append(key.encode("utf-8"))
                    scores.append(-1000.0)
                    toktypes.append(4)  # user-defined token type

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]

        if "num_key_value_heads" in self.hparams:
            head_count_kv = self.hparams["num_key_value_heads"]
        else:
            head_count_kv = head_count

        if "_name_or_path" in self.hparams:
            hf_repo = self.hparams["_name_or_path"]
        else:
            hf_repo = ""

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            print("gguf: can not find ctx length parameter.")
            sys.exit()

        self.gguf_writer.add_name(self.dir_model.name)
        self.gguf_writer.add_source_hf_repo(hf_repo)
        self.gguf_writer.add_tensor_data_layout("Meta AI original pth")
        self.gguf_writer.add_context_length(ctx_length)
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

        if "rope_scaling" in self.hparams and self.hparams["rope_scaling"] is not None and "factor" in self.hparams["rope_scaling"]:
            if "type" in self.hparams["rope_scaling"]:
                if self.hparams["rope_scaling"]["type"] == "linear":
                    self.gguf_writer.add_rope_scale_linear(self.hparams["rope_scaling"]["factor"])

    def _reverse_hf_permute(self, weights: NDArray, n_head: int, n_kv_head: int | None = None) -> NDArray:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def _reverse_hf_permute_part(self, weights: NDArray, n_part: int, n_head: int, n_head_kv: int | None = None) -> NDArray:
        r = weights.shape[0] // 3
        return (self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv))

    def _reverse_hf_part(self, weights: NDArray, n_part: int) -> NDArray:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        if "num_key_value_heads" in self.hparams:
            head_count_kv = self.hparams["num_key_value_heads"]
        else:
            head_count_kv = head_count

        for i in range(block_count):
            if f"model.layers.{i}.self_attn.W_pack.weight" in model_kv:
                print(f"Unpacking and permuting layer {i}")
                model_kv[f"model.layers.{i}.self_attn.q_proj.weight"] = self._reverse_hf_permute_part(
                    model_kv[f"model.layers.{i}.self_attn.W_pack.weight"], 0, head_count, head_count)
                model_kv[f"model.layers.{i}.self_attn.k_proj.weight"] = self._reverse_hf_permute_part(
                    model_kv[f"model.layers.{i}.self_attn.W_pack.weight"], 1, head_count, head_count_kv)
                model_kv[f"model.layers.{i}.self_attn.v_proj.weight"] = self._reverse_hf_part(
                    model_kv[f"model.layers.{i}.self_attn.W_pack.weight"], 2)
                del model_kv[f"model.layers.{i}.self_attn.W_pack.weight"]

        for name, data in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            data = data.squeeze().numpy()

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

            print(name + " -> " + new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))
            self.gguf_writer.add_tensor(new_name, data)


class FalconModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_writer.add_name("Falcon")
        self.gguf_writer.add_context_length(2048)  # not in config.json
        self.gguf_writer.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_writer.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        head_dim = self.hparams["hidden_size"] // n_head
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data in self.get_tensors():
            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            # QKV tensor transform
            # The original query_key_value tensor contains n_head_kv "kv groups",
            # each consisting of n_head/n_head_kv query weights followed by one key
            # and one value weight (shared by all query heads in the kv group).
            # This layout makes it a big pain to work with in GGML.
            # So we rearrange them here,, so that we have n_head query weights
            # followed by n_head_kv key weights followed by n_head_kv value weights,
            # in contiguous fashion.
            # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

            if "query_key_value" in name:
                qkv = data.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
                q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                data = torch.cat((q, k, v)).reshape_as(data)

            data = data.squeeze().numpy()

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

            print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

            self.gguf_writer.add_tensor(new_name, data)


class StarCoderModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("StarCoder")
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


class RefactModel(Model):
    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        self.gguf_writer.add_name("Refact")
# refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])

        self.gguf_writer.add_feed_forward_length(ff_dim)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)

    def write_tensors(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head
        block_count = self.hparams["n_layer"]

        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        tensors = dict(self.get_tensors())
        for i in range(block_count):
            if f"transformer.h.{i}.attn.kv.weight" in tensors:
                data = tensors[f"transformer.h.{i}.attn.kv.weight"]
                tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = data[
                    : n_head_kv * head_dim
                ]
                tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = data[
                    n_head_kv * head_dim:
                ]
                del tensors[f"transformer.h.{i}.attn.kv.weight"]
            if f"transformer.h.{i}.attn.q.weight" in tensors:
                tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = tensors[
                    f"transformer.h.{i}.attn.q.weight"
                ]
                del tensors[f"transformer.h.{i}.attn.q.weight"]
            if f"transformer.h.{i}.mlp.gate_up_proj.weight" in tensors:
                data = tensors[f"transformer.h.{i}.mlp.gate_up_proj.weight"]
                tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = data[:ff_dim]
                tensors[f"model.layers.{i}.mlp.up_proj.weight"] = data[ff_dim:]
                del tensors[f"transformer.h.{i}.mlp.gate_up_proj.weight"]

        for name, data in tensors.items():
            old_dtype = data.dtype

            # convert any unsupported data types to float32
            if data.dtype != torch.float16 and data.dtype != torch.float32:
                data = data.to(torch.float32)

            data = data.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight",))
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


class PersimmonModel(Model):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = head_count
        hidden_size = self.hparams["hidden_size"]

        self.gguf_writer.add_name('persimmon-8b-chat')
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_feed_forward_length(self.hparams["ffn_hidden_size"])
        self.gguf_writer.add_rope_dimension_count(hidden_size // head_count)
        self.gguf_writer.add_head_count(head_count)
        self.gguf_writer.add_head_count_kv(head_count_kv)
        self.gguf_writer.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layernorm_epsilon"])

    def set_vocab(self):
        tokens, scores, toktypes = self._get_sentencepiece_tokenizer_info()
        self.gguf_writer.add_tokenizer_model('llama')
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_bos_token_id(71013)
        # self.gguf_writer.add_eos_token_id(71013)

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        print(tensor_map)
        for name, data in self.get_tensors():
            if name.endswith(".self_attention.rotary_emb.inv_freq"):
                continue
            old_dtype = data.dtype
            # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
            data = data.to(torch.float32).squeeze().numpy()
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print("Can not map tensor '" + name + "'")
                sys.exit()
            n_dims = len(data.shape)
            print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))
            self.gguf_writer.add_tensor(new_name, data)

    def _get_sentencepiece_tokenizer_info(self):
        from sentencepiece import SentencePieceProcessor
        tokenizer_path = self.dir_model / 'tokenizer.model'
        tokenizer = SentencePieceProcessor(str(tokenizer_path))

        print('gguf: getting sentencepiece tokenizer from', tokenizer_path)
        print('gguf: adding tokens')
        tokens: list[bytes] = []
        scores: list[float] = []
        toktypes: list[int] = []

        for i in range(tokenizer.vocab_size()):
            text: bytes
            score: float

            piece = tokenizer.id_to_piece(i)
            text = piece.encode("utf-8")
            score = tokenizer.get_score(i)

            toktype = 1
            if tokenizer.is_unknown(i):
                toktype = 2
            if tokenizer.is_control(i):
                toktype = 3
            if tokenizer.is_unused(i):
                toktype = 5
            if tokenizer.is_byte(i):
                toktype = 6

            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
            pass
        return tokens, scores, toktypes
