import json
with open("rwkv_orig_vocab.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)
    s = ""
    with open("rwkv_vocab.embd", "w", encoding="utf-8") as f2:
        for key in encoder:
            # key = key.replace("\\","\\\\")
            # key = key.replace("\"","\\\"")
            # s += "\""+key+"\",\n"
            s += key +"\n"
        f2.write(s)

print("OK")