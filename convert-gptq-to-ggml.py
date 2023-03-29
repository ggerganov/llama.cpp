# Convert a GPTQ quantized LLaMA model to a ggml compatible file
# Based on: https://github.com/qwopqwop200/GPTQ-for-LLaMa
#
import os
import re
import sys
import json
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

if len(sys.argv) != 4:
    print("Usage: convert-gptq-to-ggml.py llamaXXb-4bit.pt tokenizer.model out.bin\n")
    sys.exit(1)

fname_model = sys.argv[1]
fname_tokenizer = sys.argv[2]
dir_out = sys.argv[3]

model = torch.load(fname_model, map_location="cpu")

n_vocab, n_embd = model['model.embed_tokens.weight'].shape
n_layer = 1 + max(int(m.group(1)) for name in model
                  if (m := re.match(r'model\.layers\.([0-9]+)', name)))

# hardcoded:
n_mult = 256
n_head = {32: 32, 40: 40, 60: 52, 80: 64}[n_layer]

tokenizer = SentencePieceProcessor(fname_tokenizer)

assert tokenizer.vocab_size() == n_vocab

fname_out = sys.argv[3]

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d66)) # magic: ggmf in hex
fout.write(struct.pack("i", 1)) # file version
fout.write(struct.pack("i", n_vocab))
fout.write(struct.pack("i", n_embd))
fout.write(struct.pack("i", n_mult))
fout.write(struct.pack("i", n_head))
fout.write(struct.pack("i", n_layer))
fout.write(struct.pack("i", n_embd // n_head)) # rot (obsolete)
fout.write(struct.pack("i", 4))


# This loop unchanged from convert-pth-to-ggml.py:
for i in range(tokenizer.vocab_size()):
    if tokenizer.is_unknown(i):
        text = " \u2047 ".encode()
    elif tokenizer.is_control(i):
        text = b""
    elif tokenizer.is_byte(i):
        piece = tokenizer.id_to_piece(i)
        if len(piece) != 6:
            print(f"Invalid token: {piece}")
            sys.exit(1)
        byte_value = int(piece[3:-1], 16)
        text = struct.pack("B", byte_value)
    else:
        text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode()
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    fout.write(struct.pack("f", tokenizer.get_score(i)))

def write_header(shape, dst_name, ftype_cur):
    sname = dst_name.encode()
    fout.write(struct.pack("iii", len(shape), len(sname), ftype_cur))
    fout.write(struct.pack("i" * len(shape), *shape[::-1]))
    fout.write(sname)

    # ensure tensor data is aligned
    tensor_data_offset = fout.tell()
    tensor_data_offset = (tensor_data_offset + 31) & -32
    fout.seek(tensor_data_offset)

def convert_non_q4(src_name, dst_name):
    v = model[src_name]
    shape = v.shape
    print(f"Processing non-Q4 variable: {src_name} with shape: {shape} and type: {v.dtype}")
    if len(shape) == 1:
        print("  Converting to float32")
        v = v.to(torch.float32)

    ftype_cur = {torch.float16: 1, torch.float32: 0}[v.dtype]

    # header
    write_header(shape, dst_name, ftype_cur)

    # data
    v.numpy().tofile(fout)

def convert_q4(src_name, dst_name, permute=False):
    zeros = model[f"{src_name}.zeros"].numpy()
    scales = model[f"{src_name}.scales"].numpy()
    bias = model[f"{src_name}.bias"].numpy()
    qweight = model[f"{src_name}.qweight"].numpy().T # transpose

    # Q4_1 does not support bias; good thing the bias is always all zeros.
    assert not np.any(bias)

    # Each int32 item is actually 8 int4 items packed together, and it's transposed.
    shape = (qweight.shape[0], qweight.shape[1] * 8)

    print(f"Processing Q4 variable: {src_name} with shape: {shape}")

    # The output format has the int4 weights in groups of 32 rather than 8.
    # It looks like this:
    # For each row:
    #   For each group of 32 columns:
    #     - addend (float32, 4 bytes)
    #     - scale (float32, 4 bytes)
    #     - weights (int4 * 32, 16 bytes)
    # Note that in the input, the scales and addends are shared between all
    # the columns in a row, so we end up wasting quite a bit of memory with
    # repeated scales and addends.

    addends = -zeros # flip sign

    # Since the output format is mixed between integers and floats, we have
    # to hackily view the floats as int32s just so numpy will let us
    # concatenate them.
    addends_view = addends.view(dtype=np.int32)
    scales_view = scales.view(dtype=np.int32)

    # Split into groups of 4 columns (i.e. 32 columns of quantized data):
    grouped = qweight.reshape([qweight.shape[0], qweight.shape[1] // 4, 4])

    # Repeat addends and scales:
    addends_rep = np.atleast_3d(addends_view).repeat(grouped.shape[1], axis=1)
    scales_rep = np.atleast_3d(scales_view).repeat(grouped.shape[1], axis=1)

    blob = np.concatenate([scales_rep, addends_rep, grouped], axis=2, casting='no')

    if permute:
        # Permute some rows to undo the permutation done by convert_llama_weights_to_hf.py.
        # This can be done after the above conversion because it doesn't affect column order/layout.
        blob = (blob.reshape(n_head, 2, shape[0] // n_head // 2, *blob.shape[1:])
                    .swapaxes(1, 2)
                    .reshape(blob.shape))

    # header
    write_header(shape, dst_name, 3) # ftype = Q4_1

    # data
    blob.tofile(fout)

convert_non_q4("model.embed_tokens.weight", "tok_embeddings.weight")
convert_non_q4("model.norm.weight", "norm.weight")
convert_non_q4("lm_head.weight", "output.weight")

for i in range(n_layer):
    convert_q4(f"model.layers.{i}.self_attn.q_proj", f"layers.{i}.attention.wq.weight", permute=True)
    convert_q4(f"model.layers.{i}.self_attn.k_proj", f"layers.{i}.attention.wk.weight", permute=True)
    convert_q4(f"model.layers.{i}.self_attn.v_proj", f"layers.{i}.attention.wv.weight")
    convert_q4(f"model.layers.{i}.self_attn.o_proj", f"layers.{i}.attention.wo.weight")

    convert_q4(f"model.layers.{i}.mlp.gate_proj", f"layers.{i}.feed_forward.w1.weight")
    convert_q4(f"model.layers.{i}.mlp.down_proj", f"layers.{i}.feed_forward.w2.weight")
    convert_q4(f"model.layers.{i}.mlp.up_proj",   f"layers.{i}.feed_forward.w3.weight")

    convert_non_q4(f"model.layers.{i}.input_layernorm.weight", f"layers.{i}.attention_norm.weight")
    convert_non_q4(f"model.layers.{i}.post_attention_layernorm.weight", f"layers.{i}.ffn_norm.weight")


fout.close()

print(f"Done. Output file: {fname_out}")
print()
