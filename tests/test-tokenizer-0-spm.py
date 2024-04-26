# tests with SPM tokenizer
#
# sample usage:
#
#   python3 tests/test-tokenizer-0-spm.py ~/Data/huggingface/Llama-2-7b-hf/
#   python3 tests/test-tokenizer-0-spm.py ~/Data/huggingface/CodeLlama-34b-Instruct-hf/
#


import argparse

from sentencepiece import SentencePieceProcessor

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
parser.add_argument("--fname-tok",   help="path to a text file to tokenize")
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
    "\n\n",
    "\n\n\n",
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
    " (",
    "\n =",
    "' era",
    "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
    "3",
    "33",
    "333",
    "3333",
    "33333",
    "333333",
    "3333333",
    "33333333",
    "333333333",
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
        res = tokenizer.encode(s, add_bos=True)
        # write to file
        with open(fname_out, 'w', encoding='utf-8') as f:
            for x in res:
                f.write(str(x) + ' \'' + tokenizer.decode(x) + '\'\n')
        print('len(res): ', len(res))
        print('len(lines): ', len(lines))
    print('results written to: ', fname_out)
