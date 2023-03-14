# Convert a LLaMA model checkpoint to a ggml compatible file
#
# Load the model using Torch
# Iterate over all variables and write them to a binary file.
#
# For each variable, write the following:
#   - Number of dimensions (int)
#   - Name length (int)
#   - Dimensions (int[n_dims])
#   - Name (char[name_length])
#   - Data (float[n_dims])
#
# By default, the bigger matrices are converted to 16-bit floats.
# This can be disabled by adding the "use-f32" CLI argument.
#
# At the start of the ggml file we write the model parameters
# and vocabulary.
#

import sys
import json
import struct
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor

if len(sys.argv) < 3:
    print("Usage: convert-ckpt-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]

fname_hparams   = sys.argv[1] + "/params.json"
fname_tokenizer = sys.argv[1] + "/../tokenizer.model"

def get_n_parts(dim):
    if dim == 4096:
        return 1
    elif dim == 5120:
        return 2
    elif dim == 6656:
        return 4
    elif dim == 8192:
        return 8
    else:
        print("Invalid dim: " + str(dim))
        sys.exit(1)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"

with open(fname_hparams, "r") as f:
    hparams = json.load(f)

tokenizer = SentencePieceProcessor(fname_tokenizer)

hparams.update({"vocab_size": tokenizer.vocab_size()})

n_parts = get_n_parts(hparams["dim"])

print(hparams)
print('n_parts = ', n_parts)

for p in range(n_parts):
    print('Processing part ', p)

    #fname_model = sys.argv[1] + "/consolidated.00.pth"
    fname_model = sys.argv[1] + "/consolidated.0" + str(p) + ".pth"
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"
    if (p > 0):
        fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin" + "." + str(p)

    model = torch.load(fname_model, map_location="cpu")

    fout = open(fname_out, "wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["dim"]))
    fout.write(struct.pack("i", hparams["multiple_of"]))
    fout.write(struct.pack("i", hparams["n_heads"]))
    fout.write(struct.pack("i", hparams["n_layers"]))
    fout.write(struct.pack("i", hparams["dim"] // hparams["n_heads"])) # rot (obsolete)
    fout.write(struct.pack("i", ftype))

    # Is this correct??
    for i in range(tokenizer.vocab_size()):
        if tokenizer.is_unknown(i):
            # "<unk>" token (translated as ??)
            text = " \u2047 ".encode("utf-8")
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
        elif tokenizer.is_control(i):
            # "<s>"/"</s>" tokens
            fout.write(struct.pack("i", 0))
        elif tokenizer.is_byte(i):
            # "<U+XX>" tokens (which may be invalid UTF-8)
            piece = tokenizer.id_to_piece(i)
            if len(piece) != 6:
                print("Invalid token: " + piece)
                sys.exit(1)
            byte_value = int(piece[3:-1], 16)
            fout.write(struct.pack("i", 1))
            fout.write(struct.pack("B", byte_value))
        else:
            # normal token. Uses U+2581 (LOWER ONE EIGHTH BLOCK) to represent spaces.
            text = tokenizer.id_to_piece(i).replace("\u2581", " ").encode("utf-8")
            fout.write(struct.pack("i", len(text)))
            fout.write(text)

    for k, v in model.items():
        name = k
        shape = v.shape

        # skip layers.X.attention.inner_attention.rope.freqs
        if name[-5:] == "freqs":
            continue

        print("Processing variable: " + name + " with shape: ", shape, " and type: ", v.dtype)

        #data = tf.train.load_variable(dir_model, name).squeeze()
        data = v.numpy().squeeze()
        n_dims = len(data.shape);

        # for efficiency - transpose some matrices
        # "model/h.*/attn/c_attn/w"
        # "model/h.*/attn/c_proj/w"
        # "model/h.*/mlp/c_fc/w"
        # "model/h.*/mlp/c_proj/w"
        #if name[-14:] == "/attn/c_attn/w" or \
        #   name[-14:] == "/attn/c_proj/w" or \
        #   name[-11:] == "/mlp/c_fc/w" or \
        #   name[-13:] == "/mlp/c_proj/w":
        #    print("  Transposing")
        #    data = data.transpose()

        dshape = data.shape

        # default type is fp16
        ftype_cur = 1
        if ftype == 0 or n_dims == 1:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

        # header
        sname = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
        for i in range(n_dims):
            fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
        fout.write(sname);

        # data
        data.tofile(fout)

    # I hope this deallocates the memory ..
    model = None

    fout.close()

    print("Done. Output file: " + fname_out + ", (part ", p, ")")
    print("")
