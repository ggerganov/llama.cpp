import ctypes
from ctypes import cdll, c_char_p, c_void_p, POINTER, c_float, c_int
import numpy as np
import os

libc = cdll.LoadLibrary("./libembdinput.so")
libc.sampling.restype=c_char_p
libc.create_mymodel.restype=c_void_p
libc.eval_string.argtypes=[c_void_p, c_char_p]
libc.sampling.argtypes=[c_void_p]
libc.eval_float.argtypes=[c_void_p, POINTER(c_float), c_int]


class MyModel:
    def __init__(self, args):
        argc = len(args)
        c_str = [c_char_p(i.encode()) for i in args]
        args_c = (c_char_p * argc)(*c_str)
        self.model = c_void_p(libc.create_mymodel(argc, args_c))
        self.max_tgt_len = 512
        self.print_string_eval = True

    def __del__(self):
        libc.free_mymodel(self.model)

    def eval_float(self, x):
        libc.eval_float(self.model, x.astype(np.float32).ctypes.data_as(POINTER(c_float)), x.shape[1])

    def eval_string(self, x):
        libc.eval_string(self.model, x.encode()) # c_char_p(x.encode()))
        if self.print_string_eval:
            print(x)

    def eval_token(self, x):
        libc.eval_id(self.model, x)

    def sampling(self):
        s = libc.sampling(self.model)
        return s

    def stream_generate(self, end="</s>"):
        ret = b""
        end = end.encode()
        for _ in range(self.max_tgt_len):
            tmp = self.sampling()
            ret += tmp
            yield tmp
            if ret.endswith(end):
                break

    def generate_with_print(self, end="</s>"):
        ret = b""
        for i in self.stream_generate(end=end):
            ret += i
            print(i.decode(errors="replace"), end="", flush=True)
        print("")
        return ret.decode(errors="replace")


    def generate(self, end="</s>"):
        text = b"".join(self.stream_generate(end=end))
        return text.decode(errors="replace")

if __name__ == "__main__":
    model = MyModel(["main", "--model", "../llama.cpp/models/ggml-vic13b-q4_1.bin", "-c", "2048"])
    model.eval_string("""user: what is the color of the flag of UN?""")
    x = np.random.random((5120,10))# , dtype=np.float32)
    model.eval_float(x)
    model.eval_string("""assistant:""")
    for i in model.generate():
        print(i.decode(errors="replace"), end="", flush=True)
