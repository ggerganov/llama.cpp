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
import argparse

from sentencepiece import SentencePieceProcessor

ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model to convert")
ARG_PARSER.add_argument("--ftype",
                        type=str,
                        required=True,
                        choices=["f16", "f32"],
                        help="Either f16 or f32")
ARG_PARSER.add_argument("--output",
                        type=str,
                        required=True,
                        help="Model to write")
ARGS = ARG_PARSER.parse_args()

FTYPE_IDX = -1
if ARGS.ftype == "f16":
    FTYPE_IDX = 1
elif ARGS.ftype == "f32":
    FTYPE_IDX = 0

fname_hparams   = ARGS.model + "/params.json"
fname_tokenizer = ARGS.model + "/../tokenizer.model"

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

fname_out = ARGS.output + "/ggml-model-" + ARGS.ftype + ".bin"

with open(fname_hparams, "r") as f:
    hparams = json.load(f)

tokenizer = SentencePieceProcessor(fname_tokenizer)

hparams.update({"vocab_size": tokenizer.vocab_size()})

n_parts = get_n_parts(hparams["dim"])

print(hparams)
print('n_parts = ', n_parts)

for p in range(n_parts):
    print('Processing part ', p)
    fname_out = ARGS.output + "/ggml-model-" + ARGS.ftype + ".bin"

    if (p > 0):
        fname_out = ARGS.output + "/ggml-model-" + ARGS.ftype + ".bin" + "." + str(p)

    model = torch.load(
        ARGS.model + "/consolidated.0" + str(p) + ".pth", map_location="cpu"
    )

    fout = open(fname_out, "wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["dim"]))
    fout.write(struct.pack("i", hparams["multiple_of"]))
    fout.write(struct.pack("i", hparams["n_heads"]))
    fout.write(struct.pack("i", hparams["n_layers"]))
    fout.write(struct.pack("i", hparams["dim"] // hparams["n_heads"])) # rot (obsolete)
    fout.write(struct.pack("i", FTYPE_IDX))

    # Is this correct??
    for i in range(32000):
        # TODO: this is probably wrong - not sure how this tokenizer works
        text = tokenizer.decode([29889, i]).encode('utf-8')
        # remove the first byte (it's always '.')
        text = text[1:]
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    for k, v in model.items():
        name = k
        shape = v.shape

        # skip layers.X.attention.inner_attention.rope.freqs
        if name[-5:] == "freqs":
            continue

        print("Processing variable: " + name + " with shape: ", shape, " and type: ", v.dtype)

        #data = tf.train.load_variable(ARGS.output, name).squeeze()
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

        if ARGS.ftype == "f32" or n_dims == 1:
            print("  Converting to float32")
            data = data.astype(np.float32)

        # header
        sname = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(sname), FTYPE_IDX))
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
