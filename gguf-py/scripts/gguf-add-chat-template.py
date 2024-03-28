import sys
import struct
import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUF_DEFAULT_ALIGNMENT


# add chat template to existed gguf and retrun new file path
def add_chat_template_to_existed_gguf(model_path: str, chat_template: str, new_file_path: str) -> None:
    reader = GGUFReader(model_path, "r+")

    # chat template key
    CHAT_TEMPLATE = "tokenizer.chat_template"

    # generate and adjust chat template data from chat template with alignment
    alignment = GGUF_DEFAULT_ALIGNMENT
    new_align = reader.fields.get('general.alignment')
    if new_align is not None:
        alignment = new_align.parts[-1][0]

    add_data = bytearray()

    name_data = CHAT_TEMPLATE.encode("utf-8")
    add_data += struct.pack("Q", len(name_data))
    add_data += name_data
    add_data += struct.pack("I", GGUFValueType.STRING.value)

    # calculate the padding length
    raw_len = len(add_data) + 8 + len(chat_template)
    add_len = alignment - (raw_len % alignment)
    if add_len != 0:
        chat_template += " " * add_len

    raw_data = chat_template.encode("utf-8")
    add_data += struct.pack("Q", len(raw_data))
    add_data += raw_data

    # insert raw bytes into file
    # find insert index
    kv = reader.fields
    last_field = list(kv.values())[-1]
    insert_offset = last_field.offset

    # copy original data
    new_data = reader.data.copy()
    new_data = np.concatenate(
        (new_data[:insert_offset], add_data, new_data[insert_offset:]))

    # add kv_count
    kv_count_idx = reader.fields["GGUF.kv_count"].parts[0][0]
    new_data[kv_count_idx] += 1

    # save file
    with open(new_file_path, "wb") as file:
        file.write(new_data.tobytes())

# sample usage: python guff-add-chat-template.py "./model.gguf" "chat template" "./new_model.gguf"
def main() -> None:
    parser = argparse.ArgumentParser(description="Add chat template to existed gguf file")
    parser.add_argument("model",          type=str,            help="GGUF format model filepath")
    parser.add_argument("chat_template",  type=str,            help="chat template for this model")
    parser.add_argument("new_model",      type=str,            help="New GGUF model filepath")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    print(f'adding chat template to {args.model}')
    add_chat_template_to_existed_gguf(args.model, args.chat_template, args.new_model)
    print(f'new model {args.new_model} with chat template added successfully!')


if __name__ == '__main__':
    main()
