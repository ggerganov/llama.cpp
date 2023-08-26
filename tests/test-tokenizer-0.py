import os
import sys
import argparse

from sentencepiece import SentencePieceProcessor

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
args = parser.parse_args()

dir_tokenizer = args.dir_tokenizer

tokenizer = SentencePieceProcessor(dir_tokenizer + '/tokenizer.model')

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
    ]


for text in tests:
    print('text: ', text)
    print('\nwith bos:')
    print(tokenizer.encode(text, add_bos=True))
    print(tokenizer.decode(tokenizer.encode(text, add_bos=True)))
    print('\nwithout bos:')
    print(tokenizer.encode(text, add_bos=False))
    print(tokenizer.decode(tokenizer.encode(text, add_bos=False)))

print("'" + tokenizer.id_to_piece(15043) + "'") # '_Hello'
print("'" + tokenizer.id_to_piece(29871) + "'") # '_'
print("'" + tokenizer.decode([15043]) + "'")        # 'Hello'
print("'" + tokenizer.decode([15043, 15043]) + "'") # 'Hello Hello'
print("'" + tokenizer.decode([29871, 15043]) + "'")               # ' Hello'
print("'" + tokenizer.decode([29871, 15043, 29871, 15043]) + "'") # ' Hello  Hello'

print("\n\ntests for C++:\n")
for text in tests:
    res = tokenizer.encode(text, add_bos=False)

    k = text.replace('\n', '\\n')
    k = k.replace('\t', '\\t')
    k = '"' + k + '"'
    print("{ %-24s, { " % k, end='')
    for x in res:
        print("%7d," % x, end='')
    print(" }, },")
