# tests with BPE tokenizer
#
# sample usage:
#
#   python3 tests/test-tokenizer-0-bpe.py ./models/ggml-vocab-llama-bpe.gguf ~/Data/huggingface/Meta-Llama-3-8B-Instruct/
#

import logging
import argparse
import subprocess
import random

import cffi
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("test-tokenizer-random-bpe")


class LibLlama:

    DEFAULT_PATH_LLAMA_H = "./llama.h"
    DEFAULT_PATH_LIBLLAMA = "./build/libllama.so"  # CMakeLists.txt: BUILD_SHARED_LIBS ON

    def __init__(self, path_llama_h:str=None, path_libllama:str=None):
        path_llama_h = path_llama_h or self.DEFAULT_PATH_LLAMA_H
        path_libllama = path_libllama or self.DEFAULT_PATH_LIBLLAMA
        (self.ffi, self.lib) = self._load_libllama_cffi(path_llama_h, path_libllama)
        self.lib.llama_backend_init()

    def _load_libllama_cffi(self, path_llama_h: str, path_libllama: str):
        cmd = ["gcc", "-E", "-P", "-D__restrict=", "-D__attribute__(x)=", "-D__asm__(x)=", path_llama_h]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert(res.returncode == 0)
        source = res.stdout.decode()
        ffi = cffi.FFI()
        if True:  # workarounds for pycparser
            source = "typedef struct { } __builtin_va_list;" + "\n" + source
            source = source.replace("sizeof (int)",    str(ffi.sizeof("int")))
            source = source.replace("sizeof (void *)", str(ffi.sizeof("void*")))
            source = source.replace("sizeof (size_t)", str(ffi.sizeof("size_t")))
            source = source.replace("sizeof(int32_t)", str(ffi.sizeof("int32_t")))
        ffi.cdef(source, override=True)
        lib = ffi.dlopen(path_libllama)
        return (ffi, lib)
    
    def model_default_params(self, **kwargs):
        mparams = self.lib.llama_model_default_params()
        for k, v in kwargs.items():
            setattr(mparams, k, v)
        return mparams
    
    def context_default_params(self, **kwargs):
        cparams = self.lib.llama_context_default_params()
        for k, v in kwargs.items():
            setattr(cparams, k, v)
        return cparams

class LibLlamaModel:

    def __init__(self, libllama:LibLlama, path_model:str, mparams={}, cparams={}):
        self.lib = libllama.lib
        self.ffi = libllama.ffi
        if type(mparams) == dict:
            mparams = libllama.model_default_params(**mparams)
        self.model = self.lib.llama_load_model_from_file(path_model.encode(), mparams)
        if not self.model:
            raise RuntimeError("error: failed to load model '%s'"%path_model)
        if type(cparams) == dict:
            cparams = libllama.context_default_params(**cparams)
        self.ctx = self.lib.llama_new_context_with_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("error: failed to create context for model '%s'"%path_model)
        n_tokens_max = self.lib.llama_n_ctx(self.ctx)
        self.token_ids = self.ffi.new("llama_token[]", n_tokens_max)

    def free(self):
        if self.ctx:
            self.lib.llama_free(self.ctx)
        if self.model:
            self.lib.llama_free_model(self.model)
        self.ctx = None
        self.model = None
        self.lib = None

    def tokenize(self, text:str, n_tokens_max:int=0, add_special:bool=False, parse_special:bool=False) -> list[int]:
        n_tokens_max = n_tokens_max if n_tokens_max > 0 else len(self.token_ids)
        text = text.encode("utf-8")
        num = self.lib.llama_tokenize(self.model, text, len(text), self.token_ids, n_tokens_max, add_special, parse_special)
        if num < 0:
            return []
        return list(self.token_ids[0:num])


def find_first_mismatch(ids1:list[int], ids2:list[int]):
    for i, (a,b) in enumerate(zip(ids1, ids2)):
        if a != b:
            return i
    return -1 if len(ids1) == len(ids2) else i


def test_custom_texts(model:LibLlamaModel, tokenizer:PreTrainedTokenizerBase):

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

    more_tests = [
        '\x1f-a',   # unicode_ranges_control, {0x00001C, 0x00001F}
        '¼-a',      # unicode_ranges_digit, 0x00BC
        '½-a',      # unicode_ranges_digit, 0x00BD
        '¾-a',      # unicode_ranges_digit, 0x00BE
        'a 〇b',    # unicode_ranges_digit, 0x3007
        'Ⅵ-a',     # unicode_ranges_digit, {0x00002150, 0x0000218F} // Number Forms
        '\uFEFF//', # unicode_ranges_control, 0xFEFF (BOM)
    ]

    for text in tests+more_tests:
        ids1 = model.tokenize(text, parse_special=True)
        ids2 = tokenizer.encode(text)
        logger.info(repr(text))
        if ids1 != ids2:
            logger.info(" TokenIDs: " + str(list(ids1)))
            logger.info(" Expected: " + str(list(ids2)))
            logger.info(" Index: %d" % find_first_mismatch(ids1, ids2))
            raise Exception()


def test_random_chars(model:LibLlamaModel, tokenizer:PreTrainedTokenizerBase, iterations=100):

    WHITESPACES = list(" "*20 + "\n"*5 + "\r\n"*5 + "\t"*5)
    CHARS = list(set("""
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        abcdefghijklmnopqrstuvwxyz
        ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜ
        áéíóúàèìòùâêîôûäëïöü
        .-,*/-+ª!"·$%&/()=?¿[]{}<>\\|@#~½¬~;:_
    """))
    
    logger.info("Bruteforce random chars encodings ...")
    rand = random.Random()
    for m in range(iterations):

        logger.debug("%d/%d" % (m+1,iterations))
        rand.seed(m)

        text = []
        num_words = rand.randint(300, 400)
        for i in range(num_words):
            k = rand.randint(1, 7)
            word = rand.choices(CHARS, k=k)
            space = rand.choice(WHITESPACES)
            text.append("".join(word)+space)
        text = "".join(text)

        ids1 = model.tokenize(text, parse_special=True)
        ids2 = tokenizer.encode(text)
        assert(ids1 == ids2)


def test_random_vocab_chars(model:LibLlamaModel, tokenizer:PreTrainedTokenizerBase, iterations=100):

    logger.info("Building vocab char list ...")
    vocab_ids = list(tokenizer.vocab.values())
    vocab_text = tokenizer.decode(vocab_ids)
    vocab_chars = list(set(vocab_text))
    del vocab_ids, vocab_text
    
    logger.info("Bruteforce random text encodings ...")
    rand = random.Random()
    for m in range(iterations):

        logger.debug("%d/%d" % (m+1,iterations))
        rand.seed(m)
        
        text = rand.choices(vocab_chars, k=1024)
        text = "".join(text)

        ids1 = model.tokenize(text, parse_special=True)
        ids2 = tokenizer.encode(text)
        assert(ids1 == ids2)


def test_random_vocab_tokens(model:LibLlamaModel, tokenizer:PreTrainedTokenizerBase, iterations=100):

    logger.info("Building token list ...")
    space_id = tokenizer.encode(" ")[0]
    vocab_ids = list(tokenizer.vocab.values())
    vocab_ids = list(sorted(vocab_ids + vocab_ids))
    for i in range(1, len(vocab_ids), 2):
        vocab_ids[i] = space_id
    vocab_tokens = tokenizer.decode(vocab_ids)
    vocab_tokens = vocab_tokens.split(" ")
    del vocab_ids
    
    logger.info("Checking single token encodings ...")
    for token in vocab_tokens:
        ids1 = model.tokenize(token, parse_special=True)
        ids2 = tokenizer.encode(token)
        assert(ids1 == ids2)

    logger.info("Bruteforce random text encodings ...")
    rand = random.Random()
    for m in range(iterations):

        logger.debug("%d/%d" % (m+1,iterations))
        rand.seed(m)
        
        text = []
        num_words = rand.randint(300, 400)
        for i in range(num_words):
            k = rand.randint(1, 3)
            tokens = rand.choices(vocab_tokens, k=k)
            tokens = [ t.strip(" \n\r\t") for t in tokens ]
            sep = rand.choice("     \n\r\t")
            text.append("".join(tokens) + sep)
        text = "".join(text)

        ids1 = model.tokenize(text, parse_special=True)
        ids2 = tokenizer.encode(text)
        assert(ids1 == ids2)


def test_random_bytes(model:LibLlamaModel, tokenizer:PreTrainedTokenizerBase, iterations=100):

    WHITESPACES = list(" "*20 + "\n"*5 + "\r\n"*5 + "\t"*5)

    logger.info("Bruteforce random bytes encodings ...")
    rand = random.Random()
    for m in range(iterations):

        logger.debug("%d/%d" % (m+1,iterations))
        rand.seed(m)

        text = []
        num_words = rand.randint(300, 400)
        for i in range(num_words):
            k = rand.randint(1, 8)
            word = [chr(r) for r in rand.randbytes(k) if r]
            word.append(rand.choice(WHITESPACES))
            text.append("".join(word))
        text = "".join(text)

        ids1 = model.tokenize(text, parse_special=True)
        ids2 = tokenizer.encode(text)
        assert(ids1 == ids2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_file", help="path to vocab 'gguf' file")
    parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model = LibLlamaModel(LibLlama(), args.vocab_file, mparams=dict(vocab_only=True), cparams=dict(n_ctx=2048))

    tokenizer = AutoTokenizer.from_pretrained(args.dir_tokenizer)

    test_custom_texts(model, tokenizer)
    test_random_chars(model, tokenizer, 10_000)
    test_random_vocab_chars(model, tokenizer, 10_000)
    test_random_vocab_tokens(model, tokenizer, 10_000)
    #test_random_bytes(model, tokenizer, 10_000)  # FAIL

    model.free()
