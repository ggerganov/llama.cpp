# Convert a LLaMA model checkpoint to a ggml compatible file
#
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

from collections import defaultdict
import sys
import json
import struct
import numpy as np
from tqdm import tqdm
import zipfile
import pickle
import concurrent.futures
import io
import threading
import queue

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

print(f'Model params.json: {hparams}')
print(f'Parts to process: {n_parts}')


def load_model(fname):
    class Tensor():
        def __init__(self, shape, dtype, loadinfo):
            self.shape = shape
            self.dtype = dtype
            self.loadinfo = loadinfo

        def numpy(self):
            myzip, base_name, storage_offset, k, shape, dtype = self.loadinfo
            with myzip.open(f'{base_name}/data/{k}') as myfile:
                bytes_size = np.dtype(self.dtype).itemsize
                myfile.seek(storage_offset * bytes_size, 1)
                ret = np.empty(shape, dtype=dtype)
                myfile.readinto(ret.data)
                return ret

    def my_unpickle(datapkl, myzip, base_name):
        def my_rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
            storage_type = storage[1]
            obj_key = storage[2]
            return Tensor(shape=size, dtype=storage_type, loadinfo=(
                myzip, base_name, storage_offset,
                obj_key, size, storage_type
            ))

        class MyUnpickler(pickle.Unpickler):
            def find_class(self, *p):
                if p == ('torch', 'HalfStorage'): return np.float16
                if p == ('torch', 'FloatStorage'): return np.float32
                if p == ('torch._utils', '_rebuild_tensor_v2'): return my_rebuild_tensor
                if p == ('collections', 'OrderedDict'): return dict
                raise ValueError(f'Unrecognized pickle {p}')

            def persistent_load(self, pid):
                return pid

        return MyUnpickler(datapkl).load()

    myzip =  zipfile.ZipFile(fname, 'r')
    base_name = myzip.namelist()[0].split('/', 1)[0]
    with myzip.open(f'{base_name}/data.pkl') as myfile:
        model = my_unpickle(myfile, myzip, base_name)
    return model

def get_fname(p):
    fname = "/consolidated.0" + str(p) + ".pth"
    return fname

def process_part(p):
    fname = get_fname(p)
    fname_model = sys.argv[1] + fname
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"
    if (p > 0):
        fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin" + "." + str(p)

    print(f"Processing part {fname}")

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

    model = load_model(fname_model)

    q = queue.Queue(maxsize=2)

    def writer():
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            fout.write(item.getvalue())
            q.task_done()

    threading.Thread(target=writer, daemon=True).start()

    for k, v in (t := tqdm(model.items(), bar_format="{r_bar} {percentage:3.0f}% |{bar:50} | {desc}")):
        t.set_description(f"Processing {k} with shape {tuple(v.shape)} and type {np.dtype(v.dtype)}")
        name = k
        shape = v.shape

        # skip layers.X.attention.inner_attention.rope.freqs
        if name[-5:] == "freqs":
            continue

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
        if ftype == 0 or n_dims == 1:
            # print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

        memout = io.BytesIO()
        # header
        sname = name.encode('utf-8')
        memout.write(struct.pack("iii", n_dims, len(sname), ftype_cur))
        for i in range(n_dims):
            memout.write(struct.pack("i", dshape[n_dims - 1 - i]))
        memout.write(sname)

        # data
        memout.write(data.tobytes())
        q.put(memout)

    q.put(None)
    q.join()

    model = None

    fout.close()

    print("Done. Output file: " + fname_out + ", (part ", p, ")")

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(process_part, p) for p in range(n_parts)}
    for f in (concurrent.futures.as_completed(futures)):
        if f.exception() is not None: raise f.exception()

print("All done.")