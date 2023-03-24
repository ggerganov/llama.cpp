import struct
import mmap

import numpy as np


def open_trace(fn):
    base_header_fmt = "i" * 2
    file = open(fn, "rb")
    magic, version = struct.unpack(base_header_fmt, file.read(struct.calcsize(base_header_fmt)))
    if magic != 0x67676d74:
        raise ValueError('Invalid file magic. Must be a llama.cpp trace file')
    parser_cls = TraceParserBase._parsers.get(version)
    if parser_cls is None:
        raise ValueError(f'Unknown version {version}')
    return parser_cls(file)

class TraceParserBase:
    def __init__(self, file):
        self.file = file
        self.mmap = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = file.tell() # Skip magic and version header
        self.size = self.mmap.size()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.mmap.close()
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.size:
            raise StopIteration
        return self.parse_record()

class TraceParserV0(TraceParserBase):
    def __init__(self, file):
        super().__init__(file)
        header_fmt = 'i' # n_vocab
        self.n_vocab, = struct.unpack_from(header_fmt, self.mmap, self.pos)
        self.pos += struct.calcsize(header_fmt)

    def parse_record(self):
        pos = self.pos
        n_vocab = self.n_vocab

        header_fmt = 'i' # n_tokens
        n_tokens, = struct.unpack_from(header_fmt, self.mmap, pos)
        pos += struct.calcsize(header_fmt)
        tokens = np.frombuffer(self.mmap, dtype=np.int32, count=n_tokens, offset=pos)
        pos += tokens.itemsize * tokens.size
        logits = np.frombuffer(self.mmap, dtype=np.float32, count=n_tokens * n_vocab, offset=pos)
        pos += logits.itemsize * logits.size

        assert pos <= self.size
        self.pos = pos
        return tokens, logits.reshape((n_tokens, n_vocab))

TraceParserBase._parsers = {
    0: TraceParserV0
}