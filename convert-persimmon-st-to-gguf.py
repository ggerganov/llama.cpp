from convert import lazy_load_safetensors_file
import sys
import torch
from safetensors import safe_open
from pathlib import Path
from pprint import pprint
from sentencepiece import SentencePieceProcessor
import argparse
import gguf
import json
import struct

def file_is_safetensors(path: Path) -> bool:
    fp = open(path, 'rb')
    first8 = fp.read(8)
    fp.seek(0)
    if first8[:2] == b'PK':
        # A zip file, i.e. PyTorch format
        return False
    return struct.unpack('<Q', first8)[0] < 16 * 1024 * 1024

def get_tokenizer_info(dir_model: Path):
    tokenizer_path = dir_model / 'adept_vocab.model'
    print('gguf: get sentencepiece tokenizer from', tokenizer_path)
    tokenizer = SentencePieceProcessor(str(tokenizer_path))  
    tokens: list[bytes] = []
    scores: list[float] = []
    toktypes: list[int] = []

    for i in range(tokenizer.vocab_size()):
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
        pass
    return tokens, scores, toktypes


def main(args_in: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a Persimmon model from Adept (e.g. Persimmon 8b chat) to a GGML compatible file")
    parser.add_argument("--dump",        action="store_true",    help="don't convert, just show what's in the model")
    parser.add_argument("--outtype",     choices=["f32"],        help="currently only support fp32")
    parser.add_argument("--outfile",     type=Path,              help="path to write to; default: based on input")
    parser.add_argument("model",         type=Path,              help="directory containing model file, or model file itself (*.safetensors)")
    parser.add_argument("--vocabtype",   choices=["spm", "bpe"], help="vocab format (default: spm)", default="spm")
    args = parser.parse_args(args_in)

    assert file_is_safetensors(args.model), 'Error: model file is not a SafeTensors file'
    dir_model = args.model.parent
    with open(dir_model / 'config.json', 'r') as f:
        hparams = json.load(f)
    pprint(hparams)
    arch = gguf.MODEL_ARCH.PERSIMMON
    gguf_writer = gguf.GGUFWriter(args.outfile, gguf.MODEL_ARCH_NAMES[arch])
    
    block_count = hparams['num_layers']
    head_count = hparams['num_attention_heads']
    head_count_kv = head_count
    ctx_length = hparams['seq_length']
    hidden_size = hparams['hidden_size']

    gguf_writer.add_name('persimmon-8b-chat')
    gguf_writer.add_context_length(ctx_length)
    gguf_writer.add_embedding_length(hidden_size)
    gguf_writer.add_block_count(block_count)
    gguf_writer.add_feed_forward_length(hparams['ffn_hidden_size'])
    gguf_writer.add_rope_dimension_count(hidden_size // head_count)
    gguf_writer.add_head_count(head_count)
    gguf_writer.add_head_count_kv(head_count_kv)
    gguf_writer.add_rope_freq_base(hparams['rotary_emb_base'])
    gguf_writer.add_layer_norm_eps(hparams['layernorm_epsilon'])
    if True:
        tokens, scores, toktypes = get_tokenizer_info(dir_model)
        gguf_writer.add_tokenizer_model('llama')
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        gguf_writer.add_token_types(toktypes)
        gguf_writer.add_bos_token_id(71013)
        gguf_writer.add_eos_token_id(71013)
    tensor_map = gguf.get_tensor_name_map(arch, block_count)
    print(tensor_map)
    tensors = {}
    with safe_open(args.model, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for name in tensors.keys():
        data = tensors[name]
        if name.endswith(".self_attention.rotary_emb.inv_freq"):
            continue
        old_dtype = data.dtype
        # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
        data = data.to(torch.float32).squeeze().numpy()
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
            print("Can not map tensor '" + name + "'")
            sys.exit()
        n_dims = len(data.shape)
        print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))

        gguf_writer.add_tensor(new_name, data)
    print("gguf: write header")
    gguf_writer.write_header_to_file()
    print("gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print(f"gguf: model successfully exported to '{args.outfile}'")
    print("")



if __name__ == '__main__':
    main()
