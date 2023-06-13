import json,os

special = []

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    global special
    special = bs
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_code_points(string):
    code_points = []
    for char in string:
        if ord(char) <= 255:
            if ord(char) in special:
                code_points.append(char)
            else:
                t = ("\\u" + format(ord(char+255), "04x"))
                code_points.append(t.decode('utf-8','ignore'))
        else:
            code_points.append("\\u" + format(ord(char), "04x"))
    return "".join(code_points)

import unicodedata

def remove_nonprintable_characters(input_string):
    cleaned_string = ''.join(
        c for c in input_string
        if unicodedata.category(c)[0] != 'C'
    )
    return cleaned_string

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
sortedbd = sorted(byte_decoder.items(), key=lambda kv: kv[1])
tr = "{"
for i in sortedbd:
    tr += "\""+i[0]+"\","
tr += "}"
print(tr)

with open((os.path.dirname(os.path.realpath(__file__))+"/") + "rwkv_world_vocab.txt", "r", encoding="utf-8") as f:
    list = f.readlines()
    s = ""
    with open("rwkv_world_vocab.embd", "w", encoding="utf-8") as f2:
        nn = 0
        for l in list:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            #dec = str(remove_nonprintable_characters(x.decode('ansi','ignore')))
            # print(str(x))
            s += x.hex() +"\n"
        f2.write(s)

print("OK")