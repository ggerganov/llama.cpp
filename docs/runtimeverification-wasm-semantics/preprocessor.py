# Preprocessor that converts Wasm concrete syntax into a form parseable by K.
# example usage: python convert.py f32.wast

import re
import sys


def hex2float(h):
    h = re.sub("_", "", h)
    if "nan" in h:
        # TODO: Keep bit pattern of float, don't turn all of them into simple NaNs.
        return re.sub("-?nan(:.*$)?", "NaN", h)
    elif "inf" in h:
        return h.replace("inf", "Infinity")
    elif "0x" in h:
        try:
            return h.split()[0] + " " + "%e" % (float.fromhex(h.split()[1]))
        except OverflowError:
            return h
    else:
        return h


def main():
    if len(list(sys.argv)) == 1:
        infile = sys.stdin
    else:
        infile = open(sys.argv[1])
    for line in (infile.readlines()):
        sys.stdout.write(re.sub(r"(?:(?:f32|f64)\.const )([^\)]+)",
                             lambda m: hex2float(m.group()),
                             line))
    infile.close()


if __name__ == "__main__":
    main()
