#!/usr/bin/env python3
import sys
from pathlib import Path

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader  # noqa: E402


def minimal_example(filename: str) -> None:
    reader = GGUFReader(filename, 'r+')
    field = reader.fields['tokenizer.ggml.bos_token_id']
    if field is None:
        return
    part_index = field.data[0]
    field.parts[part_index][0] = 2 # Set tokenizer.ggml.bos_token_id to 2
    #
    # So what's this field.data thing? It's helpful because field.parts contains
    # _every_ part of the GGUF field. For example, tokenizer.ggml.bos_token_id consists
    # of:
    #
    #  Part index 0: Key length (27)
    #  Part index 1: Key data ("tokenizer.ggml.bos_token_id")
    #  Part index 2: Field type (4, the id for GGUFValueType.UINT32)
    #  Part index 3: Field value
    #
    # Note also that each part is an NDArray slice, so even a part that
    # is only a single value like the key length will be a NDArray of
    # the key length type (numpy.uint32).
    #
    # The .data attribute in the Field is a list of relevant part indexes
    # and doesn't contain internal GGUF details like the key length part.
    # In this case, .data will be [3] - just the part index of the
    # field value itself.


def change_gguf(reader: GGUFReader, key: str, value: str) -> None:
    field = reader.get_field(key)
    if field is None:
        print(f'! Field {repr(key)} not found', file = sys.stderr)
        sys.exit(1)
    # Note that field.types is a list of types. This is because the GGUF
    # format supports arrays. For example, an array of UINT32 would
    # look like [GGUFValueType.ARRAY, GGUFValueType.UINT32]
    handler = reader.gguf_scalar_to_np.get(field.types[0]) if field.types else None
    if handler is None:
        print(f'! Field {repr(key)} has unsupported type: {field.types}')
        sys.exit(1)
    current_value = field.parts[field.data[0]][0]
    new_value = handler(value)
    print(f'* Preparing to change field {repr(key)} from {current_value} to {new_value}')
    if current_value == new_value:
        print(f'- Key {repr(key)} already set to requested value {current_value}')
        sys.exit(0)
    print('*** Warning *** Warning *** Warning **')
    print('* Changing fields in a GGUF file can damage it. If you are positive then type YES:')
    response = input('YES, I am sure> ')
    if response != 'YES':
        print("You didn't enter YES. Okay then, see ya!")
        sys.exit(0)
    field.parts[field.data[0]][0] = new_value
    print('* Field changed. Successful completion.')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            'modify_gguf: Error: Missing arguments. Syntax: modify_gguf.py <filename> <key> <value>',
            file = sys.stderr,
        )
        sys.exit(1)
    print(f'* Loading: {sys.argv[1]}')
    reader = GGUFReader(sys.argv[1], 'r+')
    change_gguf(reader, sys.argv[2], sys.argv[3])
