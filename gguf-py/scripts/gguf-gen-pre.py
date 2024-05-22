#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.huggingface_hub import HFVocabRequest

logger = logging.getLogger("gguf-gen-pre")


# NOTE: It's impossible to catch all edge cases.
# Most naive way to handle this is to a have a pre-compiled unicode list of all 1.1 million characters
# as it's finite and iso standardized.
# This means we can predict the upper bound and can apply known time complexity solutions to
# discover the best way resolve it.
def test_pre_tok_params() -> list[str]:
    return [
        "ü, ǖ, ǘ, ǚ, ǜ",  # diaeresis
        "綠, 女, 怒, 玉, 句",  # pinyin
        "ied 4 ½ months",  # ordinal
        "¡Hola Mundo!",  # spanish
        "Olá Mundo!", # portuguese
        "Selam Dünya!",  # turkish
        "Salam, dünýä!", # turkman
        "Γειά σου Κόσμε!",  # greek
        "हैलो वर्ल्ड!",  # hindi
        "สวัสดีชาวโลก!", # thai
        "こんにちは世界！",  # japanese
        "你好世界！",  # chinese
        "Hàlo a Shaoghail!",  # gaelic
        "Chào thế giới!",  # vietnamese
        "Привет, мир!", # russian
        "Здравей свят!", # bulgarian
        "សួស្តី​ពិភពលោក!",  # kymer
        "The quick brown fox jumped over the lazy dog.", # uses every letter in en alpha
        "Le rapide renard brun sauta par dessus le chien paresseux.", # french
        "\tWil je een kopje thee?\n",  # dutch
        " Te gustaría algo de té ?   ",  # spanish
        # NOTE: I expect right-to-left languages to fail
        "העלא וועלט!", # yiddish (r-to-l)
        "سلام دنیا!",  # persian (r-to-l)
        "",  # Why?; This is a falsy value in python, no symbols.
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
        "Hello, y'all! How are you 😁 局外人?苹果apple工作work3.14159天God～",
        "3",
        "33",
        "333",
        "3333",
        "33333",
        "333333",
        "3333333",
    ]


def test_pre_tok(hf_voc_req: HFVocabRequest) -> None:
    # NOTE: aggregate all models to their respective paths
    from transformers import AutoTokenizer

    params = test_pre_tok_params()
    for model in hf_voc_req.models:
        # set the model path, e.g. 'models/meta-llama/Llama-2-7b-hf'
        path = Path(f"{hf_voc_req.model_path}/{model['repo']}")
        # set the model name, e.g. llama-2-7b-hf
        name = path.stem.lower()
        # model input encodings, e.g. 'models/meta-llama/Llama-2-7b-hf/llama-2-7b-hf.vocab.gguf.inp'
        inp = path / f"ggml-vocab-{name}.inp"
        # model output encodings, e.g. 'models/meta-llama/Llama-2-7b-hf/llama-2-7b-hf.vocab.gguf.out'
        out = path / f"ggml-vocab-{name}.out"
        # extracted tokenizer model
        final = path / f"ggml-vocab-{name}.gguf"

        # skip tokenizer folder if unavailable
        if not path.exists():
            logger.warning(f"skipped - {model['repo']} not found.")
            continue

        try:  # create the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
        except OSError as e:
            logger.error(f"{model['repo']} not found: {e}")
            continue  # skip this tokenizer model

        with open(inp, "w", encoding="utf-8") as f:
            for test in params:
                f.write(f"{test}")
                f.write("\n__ggml_vocab_test__\n")

        with open(out, "w", encoding="utf-8") as f:
            for test in params:
                encodings = tokenizer.encode(test, add_special_tokens=False)
                for encoding in encodings:
                    f.write(f" {encoding}")
                f.write("\n")

        logger.info(f"Tests for {model['repo']} written in {final}.*")


def generate_vocab_script(hf_voc_req: HFVocabRequest) -> None:
    # generate commands for creating vocab files
    shscript = "#!/usr/bin/env bash\n\n"

    for model in hf_voc_req.models:
        # get the repo path
        path = Path(f"{hf_voc_req.model_path}/{model['repo']}")
        # set the vocab path
        vocab = path / f"ggml-vocab-{path.stem.lower()}.gguf"
        # set the command line
        tmpline = f"python3 convert-hf-to-gguf.py {path} --outfile {vocab} --vocab-only\n"
        shscript += tmpline
        logger.info(tmpline.strip())

    with open("generate-vocab.sh", "w", encoding="utf-8") as f:
        f.writelines(shscript)
        logger.info(f"Wrote {len(shscript)} bytes to generate-vocab.sh")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_auth_token", help="A huggingface read auth token")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity."
    )
    parser.add_argument(
        "-r", "--model-repo", default="meta-llama/Llama-2-7b-hf",
        help="The models repository. Default is 'meta-llama/Llama-2-7b-hf'."
    )
    parser.add_argument(
        "-m", "--model-path", default="models/",
        help="The models storage path. Default is 'models/'."
    )
    parser.add_argument(
        "-a", "--model-arch", default="llama",
        help="The supported model architecture. Default is 'llama'."
    )
    parser.add_argument(
        "-p", "--model-parts", default=2,
        help="The number of model shards encompassing the model. Default is 2."
    )
    parser.add_argument(
        "-t", "--model-type", default="safetensors",
        help="The models file type. Default is 'safetensors'"
    )
    parser.add_argument(
        "-b", "--vocab-type",
        default="SPM", const="SPM", nargs="?", choices=["SPM", "BPE", "WPM"],
        help="The models tokenizer type. Default is 'SPM'."
    )
    parser.add_argument(
        "-t", "--gen-vocab-tests", action="store_true",
        help="Generate the tokenizer tests. Default is False."
    )
    parser.add_argument(
        "-s", "--gen-vocab-script", action="store_true",
        help="Generate the gguf vocab files. Default is False."
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    hf_vocab_req = HFVocabRequest(
        args.model_path, args.hf_auth_token, logger
    )

    hf_vocab_req.download_models()
    hf_vocab_req.generate_checksums()
    hf_vocab_req.log_pre_tokenizer_info()

    if args.gen_vocab_tests:
        test_pre_tok(hf_vocab_req)

    if args.gen_vocab_script:
        generate_vocab_script(hf_vocab_req)


if __name__ == '__main__':
    main()
