import json,os

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
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}
sortedbd = sorted(byte_decoder.items(), key=lambda kv: kv[1])
tr = "{"
for i in sortedbd:
    tr += "\""+i[0]+"\","
tr += "}"
print(tr)

with open((os.path.dirname(os.path.realpath(__file__))+"/") + "rwkv_orig_vocab.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)
    s = ""
    with open("rwkv_vocab.embd", "w", encoding="utf-8") as f2:
        for key in encoder:
            #key = bytearray([byte_decoder[c] for c in key]).decode('utf-8','ignore')
            # key = key.replace("\\","\\\\")
            # key = key.replace("\"","\\\"")
            # s += "\""+key+"\",\n"
            s += key +"\n"            
        f2.write(s)

print("OK")