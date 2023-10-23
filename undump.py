#!/usr/bin/env python3
import struct
import numpy as np
from pathlib import Path

def undump(fn):
    with open(fn, 'rb') as df:
        dims = struct.unpack('=QQQQ', df.read(8*4))
        (dsz,) = struct.unpack('=Q', df.read(8))
        ## assume f32
        data = df.read(dsz)
        data = [i for (i,) in struct.iter_unpack('=f', data)]
        return np.array(data).reshape(dims).squeeze()

if __name__ == '__main__':
    for dfn in sorted(Path('.').glob('*.dump')):
        darr = undump(dfn)
        print(f'{dfn}: {darr.shape}\n{darr}')

