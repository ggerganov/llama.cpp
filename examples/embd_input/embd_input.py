import ctypes
from ctypes import cdll, c_char_p, c_void_p, POINTER, c_float, c_int
import numpy as np

libc = cdll.LoadLibrary("./libembd_input.so")
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
#         print("self.model", self.model)
    def __del__(self):
        libc.free_mymodel(self.model)

    def eval_float(self, x):
        libc.eval_float(self.model, x.astype(np.float32).ctypes.data_as(POINTER(c_float)), x.shape[0])

    def eval_string(self, x):
        libc.eval_string(self.model, x.encode()) # c_char_p(x.encode()))

    def eval_token(self, x):
        libc.eval_id(self.model, x)

    def sampling(self):
        s = libc.sampling(self.model)
        return s

if __name__ == "__main__":
    model = MyModel(["main", "--model", "../llama.cpp/models/ggml-vic13b-q4_1.bin", "-c", "2048"])
    # print(model)
    model.eval_string("""user: what is the color of the flag of UN?""")
    # model.eval_token(100)
    x = np.random.random((10, 5120))# , dtype=np.float32)
    model.eval_float(x)
    model.eval_string("""assistant:""")
    # print(x[0,0], x[0,1],x[1,0])
    # model.eval_float(x)
    # print(libc)

    for i in range(500):
        tmp = model.sampling().decode()
        if tmp == "":
            break
        print(tmp, end="", flush=True)
