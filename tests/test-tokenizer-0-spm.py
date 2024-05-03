# tests with SPM tokenizer
#
# sample usage:
#
#   python3 tests/test-tokenizer-0-spm.py ~/Data/huggingface/Llama-2-7b-hf/
#   python3 tests/test-tokenizer-0-spm.py ~/Data/huggingface/CodeLlama-34b-Instruct-hf/
#


import logging
import argparse

from sentencepiece import SentencePieceProcessor

logger = logging.getLogger("test-tokenizer-0-spm")

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
parser.add_argument("--fname-tok",   help="path to a text file to tokenize")
parser.add_argument("--verbose", action="store_true", help="increase output verbosity")

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

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
    message_log = (f"text: {text}\n"
                   "with bos:\n"
                   f"{tokenizer.encode(text, add_bos=True)}\n"
                   f"{tokenizer.decode(tokenizer.encode(text, add_bos=True))}\n"
                   "without bos:\n"
                   f"{tokenizer.encode(text, add_bos=False)}\n"
                   f"{tokenizer.decode(tokenizer.encode(text, add_bos=False))}\n")
    logger.info(message_log)

logger.info(f"'{tokenizer.id_to_piece(15043)}'") # '_Hello'
logger.info(f"'{tokenizer.id_to_piece(29871)}'") # '_'
logger.info(f"'{tokenizer.decode([15043])}'")        # 'Hello'
logger.info(f"'{tokenizer.decode([15043, 15043])}'") # 'Hello Hello'
logger.info(f"'{tokenizer.decode([29871, 15043])}'")               # ' Hello'
logger.info(f"'{tokenizer.decode([29871, 15043, 29871, 15043])}'") # ' Hello  Hello'

logger.info("\n\ntests for C++:\n")
for text in tests:
    res = tokenizer.encode(text, add_bos=False)

    # Modify text representation for logging
    k = text.replace('\n', '\\n')
    k = k.replace('\t', '\\t')
    k = '"' + k + '"'

    # Log the modified text and its encoding
    log_message = "{ %-24s, { " % k
    for x in res:
        log_message += "%7d," % x
    log_message += " }, },"
    logger.info(log_message)

logger.info(tokenizer.encode('hello'))
logger.info(tokenizer.encode('world'))
logger.info(tokenizer.encode(' world'))
logger.info(tokenizer.encode('hello world'))

fname_tok = args.fname_tok
if fname_tok:
    logger.info(f"tokenizing file: {fname_tok}")
    fname_out = fname_tok + '.tok'
    with open(fname_tok, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        s = ''.join(lines)
        res = tokenizer.encode(s, add_bos=True)
        # write to file
        with open(fname_out, 'w', encoding='utf-8') as f:
            for x in res:
                f.write(str(x) + ' \'' + tokenizer.decode(x) + '\'\n')
        logger.info(f"len(res): {len(res)}")
        logger.info(f"len(lines): {len(lines)}")
    logger.info(f"results written to: {fname_out}")
