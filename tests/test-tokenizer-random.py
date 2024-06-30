# Test libllama tokenizer == AutoTokenizer.
# Brute force random words/text generation.
#
# Sample usage:
#
#   python3 tests/test-tokenizer-random.py ./models/ggml-vocab-llama-bpe.gguf ./models/tokenizers/llama-bpe
#

import time
import logging
import argparse
import subprocess
import random
import unicodedata

from typing import Callable, Iterator

import cffi
from transformers import AutoTokenizer


logger = logging.getLogger("test-tokenizer-random")


class LibLlama:

    DEFAULT_PATH_LLAMA_H = "./llama.h"
    DEFAULT_PATH_LIBLLAMA = "./build/libllama.so"  # CMakeLists.txt: BUILD_SHARED_LIBS ON

    def __init__(self, path_llama_h: str = None, path_libllama: str = None):
        path_llama_h = path_llama_h or self.DEFAULT_PATH_LLAMA_H
        path_libllama = path_libllama or self.DEFAULT_PATH_LIBLLAMA
        (self.ffi, self.lib) = self._load_libllama_cffi(path_llama_h, path_libllama)
        self.lib.llama_backend_init()

    def _load_libllama_cffi(self, path_llama_h: str, path_libllama: str):
        cmd = ["gcc", "-E", "-P", "-D__restrict=", "-D__attribute__(x)=", "-D__asm__(x)=", path_llama_h]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert (res.returncode == 0)
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

    def __init__(self, libllama: LibLlama, path_model: str, mparams={}, cparams={}):
        self.lib = libllama.lib
        self.ffi = libllama.ffi
        if isinstance(mparams, dict):
            mparams = libllama.model_default_params(**mparams)
        self.model = self.lib.llama_load_model_from_file(path_model.encode(), mparams)
        if not self.model:
            raise RuntimeError("error: failed to load model '%s'" % path_model)
        if isinstance(cparams, dict):
            cparams = libllama.context_default_params(**cparams)
        self.ctx = self.lib.llama_new_context_with_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("error: failed to create context for model '%s'" % path_model)
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

    def tokenize(self, text: str, n_tokens_max: int = 0, add_special: bool = False, parse_special: bool = False) -> list[int]:
        n_tokens_max = n_tokens_max if n_tokens_max > 0 else len(self.token_ids)
        text = text.encode("utf-8")
        num = self.lib.llama_tokenize(self.model, text, len(text), self.token_ids, n_tokens_max, add_special, parse_special)
        if num < 0:
            return []
        return list(self.token_ids[0:num])


def generator_custom_text() -> Iterator[str]:
    """General tests"""
    yield from [
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


def generator_custom_text_edge_cases() -> Iterator[str]:
    """Edge cases found while debugging"""
    yield from [
        '\x1f-a',     # unicode_ranges_control, {0x00001C, 0x00001F}
        '¼-a',        # unicode_ranges_digit, 0x00BC
        '½-a',        # unicode_ranges_digit, 0x00BD
        '¾-a',        # unicode_ranges_digit, 0x00BE
        'a 〇b',      # unicode_ranges_digit, 0x3007
        'Ⅵ-a',       # unicode_ranges_digit, {0x00002150, 0x0000218F} // Number Forms
        '\uFEFF//',   # unicode_ranges_control, 0xFEFF (BOM)
        'Cửa Việt',   # llama-3, ignore_merges = true
        '<s>a',       # Phi-3 fail
        '<unk><|endoftext|><s>',  # Phi-3 fail
        'a\na',            # bert fail
        '"`',              # falcon
        ' \u2e4e',         # falcon
        'a\xa0\xa0\x00b',  # jina-v2-es
        'one <mask>',      # jina-v2-es  <mask> lstrip=true
        'a </s> b',        # rstrip phi-3
        'a <mask> b',      # lstrip jina-v2
        '\xa0aC',          # deepseek
    ]


def generator_vocab_words(vocab: list[str]) -> Iterator[str]:
    """Brute force check all vocab words"""
    yield from vocab


def generator_added_lr_strip(tokenizer) -> Iterator[str]:
    WHITESPACES = ["", " ", "  ", "    "]
    special_tokens = list(tokenizer.all_special_tokens)
    added_tokens   = list(tokenizer.added_tokens_encoder)
    all_tokens     = list(sorted(set(special_tokens + added_tokens)))
    for token in all_tokens:
        for lstrip in WHITESPACES:
            for rstrip in WHITESPACES:
                yield lstrip + token + rstrip
                yield "a" + lstrip + token + rstrip
                yield lstrip + token + rstrip + "z"
                yield "a" + lstrip + token + rstrip + "z"


def generator_random_added_tokens(tokenizer, iterations=100) -> Iterator[str]:
    special_tokens = list(tokenizer.all_special_tokens)
    added_tokens   = list(tokenizer.added_tokens_encoder)
    separations    = [" ", "\n", "\t", "-", "!", "one", "1", "<s>", "</s>"]
    all_tokens     = list(sorted(set(special_tokens + added_tokens + separations)))
    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        words = rand.choices(all_tokens, k=500)
        if words and words[0] == tokenizer.bos_token:  # skip spam warning of double BOS
            while len(words) > 1 and words[1] == tokenizer.bos_token:  # leave one starting BOS
                words.pop(0)
            if tokenizer.add_bos_token:  # drop all starting BOS
                words.pop(0)
        if words and words[-1] == tokenizer.eos_token:  # skip spam warning of double EOS
            while len(words) > 1 and words[-2] == tokenizer.eos_token:  # leave one trailing EOS
                words.pop(-1)
            if tokenizer.add_bos_token:  # drop all trailing EOS
                words.pop(-1)
        yield "".join(words)


def generator_random_chars(iterations=100) -> Iterator[str]:
    """Brute force random text with simple characters"""

    NUM_WORDS = 400
    WHITESPACES = list(" " * 20 + "\n" * 5 + "\r\n" * 5 + "\t" * 5)
    CHARS = list(sorted(set("""
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        abcdefghijklmnopqrstuvwxyz
        ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜ
        áéíóúàèìòùâêîôûäëïöü
        .-,*/-+ª!"·$%&/()=?¿[]{}<>\\|@#~½¬~;:_
    """)))

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        for _ in range(NUM_WORDS):
            k = rand.randint(1, 7)
            word = rand.choices(CHARS, k=k)
            word.append(rand.choice(WHITESPACES))
            text.append("".join(word))
        yield "".join(text)


def generator_unicodes() -> Iterator[str]:
    """Iterate unicode characters"""

    MAX_CODEPOINTS = 0x30000  # 0x110000

    def _valid(cpt):
        if cpt >= 0x30000:  # unassigned and supplement­ary
            return False
        if 0x00D800 <= cpt <= 0x00F8FF:  # Surrogates
            return False
        if unicodedata.category(chr(cpt)) == "Cn":
            return False
        return True

    characters = [chr(cpt) for cpt in range(1, MAX_CODEPOINTS) if _valid(cpt)]

    yield from characters


def generator_random_unicodes(iterations=100) -> Iterator[str]:
    """Brute force random text with unicode characters"""

    NUM_WORDS = 200
    WHITESPACES = list(" " * 20 + "\n" * 5 + "\r\n" * 5 + "\t" * 5)

    characters = list(generator_unicodes())

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        for _ in range(NUM_WORDS):
            k = rand.randint(1, 7)
            word = rand.choices(characters, k=k)
            word.append(rand.choice(WHITESPACES))
            text.append("".join(word))
        yield "".join(text)


def generator_random_vocab_chars(vocab: list[str], iterations=100) -> Iterator[str]:
    """Brute force random text with vocab characters"""

    vocab_chars = set()
    for word in vocab:
        vocab_chars.update(word)
    vocab_chars = list(sorted(vocab_chars))

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = rand.choices(vocab_chars, k=1024)
        yield "".join(text)


def generator_random_vocab_words(vocab: list[str], iterations=100) -> Iterator[str]:
    """Brute force random text from vocab words"""

    vocab = [w.strip() for w in vocab]
    yield from vocab

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        num_words = rand.randint(300, 400)
        for i in range(num_words):
            k = rand.randint(1, 3)
            words = rand.choices(vocab, k=k)
            sep = rand.choice("     \n\r\t")
            text.append("".join(words) + sep)
        yield "".join(text)


def compare_tokenizers(func_tokenize1: Callable, func_tokenize2: Callable, generator: Iterator[str]):

    def find_first_mismatch(ids1: list[int], ids2: list[int]):
        for i, (a, b) in enumerate(zip(ids1, ids2)):
            if a != b:
                return i
        if len(ids1) == len(ids2):
            return -1
        return min(len(ids1), len(ids2))

    t_tokenizer1 = 0
    t_tokenizer2 = 0
    t_start = time.perf_counter()
    num_errors = 10

    logger.info("%s: %s" % (generator.__name__, "ini"))
    for text in generator:
        # print(repr(text), hex(ord(text[0])), text.encode())
        t0 = time.perf_counter()
        ids1 = func_tokenize1(text)
        t1 = time.perf_counter()
        ids2 = func_tokenize2(text)
        t2 = time.perf_counter()
        t_tokenizer1 += t1 - t0
        t_tokenizer2 += t2 - t1
        if ids1 != ids2:
            i = find_first_mismatch(ids1, ids2)
            ids1 = list(ids1)[max(0, i - 2) : i + 5 + 1]
            ids2 = list(ids2)[max(0, i - 2) : i + 5 + 1]
            logger.error(" TokenIDs: " + str(ids1))
            logger.error(" Expected: " + str(ids2))
            # raise Exception()
            num_errors += 1
            if num_errors > 10:
                break

    t_total = time.perf_counter() - t_start
    logger.info("%s: end,  tok1: %.3f  tok2: %.3f  total: %.3f" % (generator.__name__, t_tokenizer1, t_tokenizer2, t_total))


def main(argv: list[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_file", help="path to vocab 'gguf' file")
    parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(argv)

    logging.basicConfig(level = logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"VOCABFILE: '{args.vocab_file}'")

    model = LibLlamaModel(LibLlama(), args.vocab_file, mparams=dict(vocab_only=True), cparams=dict(n_ctx=4096))
    tokenizer = AutoTokenizer.from_pretrained(args.dir_tokenizer)

    def func_tokenize1(text: str):
        return model.tokenize(text, add_special=True, parse_special=True)

    def func_tokenize2(text: str):
        return tokenizer.encode(text, add_special_tokens=True)

    ids = func_tokenize2("a")
    assert 1 <= len(ids) <= 3
    add_bos_token = len(ids) > 1 and tokenizer.bos_token_id == ids[0]
    add_eos_token = len(ids) > 1 and tokenizer.eos_token_id == ids[-1]
    tokenizer.add_bos_token = getattr(tokenizer, "add_bos_token", add_bos_token)
    tokenizer.add_eos_token = getattr(tokenizer, "add_eos_token", add_eos_token)

    vocab = list(sorted(tokenizer.batch_decode(list(tokenizer.get_vocab().values()), skip_special_tokens=True)))

    compare_tokenizers(func_tokenize1, func_tokenize2, generator_custom_text())
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_custom_text_edge_cases())
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_unicodes())
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_vocab_words(vocab))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_added_lr_strip(tokenizer))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_random_added_tokens(tokenizer, 10_000))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_random_chars(10_000))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_random_unicodes(10_000))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_random_vocab_chars(vocab, 10_000))
    compare_tokenizers(func_tokenize1, func_tokenize2, generator_random_vocab_words(vocab, 5_000))

    model.free()


if __name__ == "__main__":
    # main()

    logging.basicConfig(
        level    = logging.DEBUG,
        format   = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        filename = logger.name + ".log",
        filemode = "a"
    )

    path_tokenizers   = "./models/tokenizers/"
    path_vocab_format = "./models/ggml-vocab-%s.gguf"

    # import os
    # tokenizers = os.listdir(path_tokenizers)
    tokenizers = [
        # "llama-spm",   # SPM
        # "phi-3",       # SPM
        # "bert-bge",    # WPM
        # "jina-v2-en",  # WPM
        "gpt-2",          # BPE
        "llama-bpe",      # BPE
        "falcon",         # BPE
        "starcoder",      # BPE
        "jina-v2-es",     # BPE
        "jina-v2-de",     # BPE
        "jina-v2-code",   # BPE
        "smaug-bpe",      # BPE
        "phi-2",          # BPE
        "deepseek-coder", # BPE
        "deepseek-llm",   # BPE
    ]

    for tokenizer in tokenizers:
        logger.info("=" * 50)
        logger.info(f"TOKENIZER: '{tokenizer}'")
        vocab_file = path_vocab_format % tokenizer
        dir_tokenizer = path_tokenizers + "/" + tokenizer
        main([vocab_file, dir_tokenizer, "--verbose"])
