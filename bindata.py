import numpy as np
import glob
import struct

def read_blck(fi):
    for x in range(1000):
        lenb = fi.read(4)
        if len(lenb)<4:
            print('error')
            return
        lena = struct.unpack('i',lenb)
        by = int(lena[0])*4
        if by <0:
            print("erro",by)
            lenb = fi.read(4)
            return
        else:
            floata = fi.read(by)
            if len(floata) == by:            
                fl = struct.unpack(f'{lena[0]}f',floata)
                #print(fl)
                if len(fl)> 4096:
                    aa = np.array(fl)
                    A = aa[:4096]
                    yield A
            else:
                print("erro",by,len(floata))

def fit_generator():
    for f in glob.glob("batch*.bin"):
        with open(f,"rb") as fi:
            one= next(read_blck(fi))
            two= next(read_blck(fi))
            print("DEBUG",one,two)
            yield one,two
