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
import os

from sentencepiece import SentencePieceProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ckpt models to ggml models.')
    parser.add_argument('dir_model',
                        type=str,
                        help='Directory path of the checkpoint model')
    parser.add_argument('ftype',
                        type=str,
                        choices=['f32', 'f16'],
                        help='Data type of the converted tensor, f32 or f16')
    parser.add_argument('out_dir',
                        type=str,
                        help='Directory path for storing ggml model')
    return parser.parse_args()


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


def main():
    args = parse_args()
    dir_model = args.dir_model
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ftype = args.ftype
    ftype_int = {'f32': 0, 'f16': 1}
    fname_hparams = os.path.join(dir_model, 'params.json')
    fname_tokenizer = os.path.join(dir_model, '..', 'tokenizer.model')

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
        fname_model = os.path.join(dir_model, "consolidated.0{}.pth".format(p))
        if p > 0:
            fname_out = os.path.join(out_dir,
                                     "ggml-model-{}.bin.{}".format(ftype, p))
        else:
            fname_out = os.path.join(out_dir,
                                     "ggml-model-{}.bin".format(ftype))

        model = torch.load(fname_model, map_location="cpu")

        fout = open(fname_out, "wb")

        fout.write(struct.pack("i", 0x67676d6c))  # magic: ggml in hex
        fout.write(struct.pack("i", hparams["vocab_size"]))
        fout.write(struct.pack("i", hparams["dim"]))
        fout.write(struct.pack("i", hparams["multiple_of"]))
        fout.write(struct.pack("i", hparams["n_heads"]))
        fout.write(struct.pack("i", hparams["n_layers"]))
        fout.write(struct.pack("i", hparams["dim"] //
                               hparams["n_heads"]))  # rot (obsolete)
        fout.write(struct.pack("i", ftype_int[ftype]))

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
                text = tokenizer.id_to_piece(i).replace("\u2581",
                                                        " ").encode("utf-8")
                fout.write(struct.pack("i", len(text)))
                fout.write(text)

        for k, v in model.items():
            name = k
            shape = v.shape

            # skip layers.X.attention.inner_attention.rope.freqs
            if name[-5:] == "freqs":
                continue

            print("Processing variable: " + name + " with shape: ", shape,
                  " and type: ", v.dtype)

            #data = tf.train.load_variable(dir_model, name).squeeze()
            data = v.numpy().squeeze()
            n_dims = len(data.shape)

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
            if ftype == 'f32' or n_dims == 1:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

            # header
            sname = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
            for i in range(n_dims):
                fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
            fout.write(sname)

            # data
            data.tofile(fout)

        # I hope this deallocates the memory ..
        model = None

        fout.close()

        print("Done. Output file: " + fname_out + ", (part ", p, ")")
        print("")


if __name__ == '__main__':
    main()
