import os
import sys
import argparse

from sentencepiece import SentencePieceProcessor

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
args = parser.parse_args()

dir_tokenizer = args.dir_tokenizer

tokenizer = SentencePieceProcessor(dir_tokenizer + '/tokenizer.model')

text = 'Hello, world!'
print(text)
print(tokenizer.encode(text, add_bos=True))
print(tokenizer.decode(tokenizer.encode(text, add_bos=True)))
