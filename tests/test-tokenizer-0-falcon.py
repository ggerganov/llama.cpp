# tests with BPE tokenizer

import argparse

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
parser.add_argument("--fname-tok",   help="path to a text file to tokenize")
args = parser.parse_args()

dir_tokenizer = args.dir_tokenizer

tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer)

tests = [
    "",
    " ",
    "  ",
    "   ",
    "\t",
    "\n",
    "\t\n",
    "Hello world",
    " Hello world",
    "Hello World",
    " Hello World",
    " Hello World!",
    "Hello, world!",
    " Hello, world!",
    " this is 🦙.cpp",
    "w048 7tuijk dsdfhu",
    "нещо на Български",
    "កាន់តែពិសេសអាចខលចេញ",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
    "Hello",
    " Hello",
    "  Hello",
    "   Hello",
    "    Hello",
    "    Hello\n    Hello",
    "\n =",
    "' era",
]

for text in tests:
    print('text: ', text)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

print("\n\ntests for C++:\n")
for text in tests:
    res = tokenizer.encode(text)

    k = text.replace('\n', '\\n')
    k = k.replace('\t', '\\t')
    k = '"' + k + '"'
    print("{ %-24s, { " % k, end='')
    for x in res:
        print("%7d," % x, end='')
    print(" }, },")

print(tokenizer.encode('hello'))
print(tokenizer.encode('world'))
print(tokenizer.encode(' world'))
print(tokenizer.encode('hello world'))

fname_tok = args.fname_tok
if fname_tok:
    print('tokenizing file: ', fname_tok)
    fname_out = fname_tok + '.tok'
    with open(fname_tok, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        s = ''.join(lines)
        res = tokenizer.encode(s)
        # write to file
        with open(fname_out, 'w', encoding='utf-8') as f:
            for x in res:
                f.write(str(x) + ' \'' + tokenizer.decode(x) + '\'\n')
        print('len(res): ', len(res))
        print('len(lines): ', len(lines))
    print('results written to: ', fname_out)
