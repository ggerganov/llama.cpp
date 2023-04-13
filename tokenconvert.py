import json
import os
import tempfile
import sys
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

if len(sys.argv) < 3:
    print("Usage: python tokenconvert.py tokenizer_type token-dir [dir-output]")
    print("  tokenizer_type: The type of tokenizer (check the model information), eg: BPE, WordPiece, SentencePiece.")
    print("  token-dir: Directory of the model containing the tokenizer.json. Example: 'bigscience/bloomz-560m'")
    print("  dir-output: directory where the output file will be written, eg: ./tokenizer.model , by default writes to the same directory.")
    sys.exit(1)

tokenizer_type = sys.argv[1]
token_dir = sys.argv[2]

if len(sys.argv) < 4:
    dir_out = token_dir
else:
    dir_out = sys.argv[3]


def load_tokenizer_from_json(json_path, special_tokens_map_path, tokenizer_config_path, tokenizer_type):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    with open(special_tokens_map_path, "r", encoding="utf-8") as f:
        special_tokens_map = json.load(f)
    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    model_data = json_data["model"]
    vocab = model_data["vocab"]
    merges = model_data["merges"]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as vocab_file:
        json.dump(vocab, vocab_file)

    if tokenizer_type == "BPE":
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as merges_file:
            merges_file.write("\n".join(merges))

        tokenizer = Tokenizer(models.BPE.from_file(vocab_file.name, merges_file.name))
        os.unlink(merges_file.name)

        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

    elif tokenizer_type == "WordPiece":
        tokenizer = Tokenizer(models.WordPiece.from_file(vocab_file.name))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.WordPiece()

    elif tokenizer_type == "SentencePiece":
        tokenizer = Tokenizer(models.SentencePiece.from_file(vocab_file.name))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.decoder = decoders.SentencePiece()


    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    os.unlink(vocab_file.name)

    bos_token_id = tokenizer.token_to_id(special_tokens_map["bos_token"])
    eos_token_id = tokenizer.token_to_id(special_tokens_map["eos_token"])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{special_tokens_map['bos_token']} $A {special_tokens_map['eos_token']}",
        pair=f"{special_tokens_map['bos_token']} $A {special_tokens_map['eos_token']} {special_tokens_map['bos_token']} $B {special_tokens_map['eos_token']}",
        special_tokens=[
            (special_tokens_map["bos_token"], bos_token_id),
            (special_tokens_map["eos_token"], eos_token_id),
        ],
    )

    return tokenizer


if __name__ == "__main__":
    input_json_path = os.path.join(token_dir, "tokenizer.json")
    special_tokens_map_path = os.path.join(token_dir, "special_tokens_map.json")
    tokenizer_config_path = os.path.join(token_dir, "tokenizer_config.json")
    output_model_path = os.path.join(dir_out, "tokenizer.model")

    tokenizer = load_tokenizer_from_json(input_json_path, special_tokens_map_path, tokenizer_config_path, tokenizer_type)
    print(f"Saving.. tokenizer.model to {output_model_path}")
    tokenizer.save(output_model_path)
    print(f"Saved tokenizer.model to {output_model_path}")

