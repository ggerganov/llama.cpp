#!/usr/bin/env python3
import sys
from pathlib import Path

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader, GGUFValueType  # noqa: E402


def dump_gguf(filename: str) -> None:
    print(f'* Loading: {filename}')
    reader = GGUFReader(filename, 'r')
    print(f'\n* Dumping {len(reader.fields)} key/value pair(s)')
    for n, field in enumerate(reader.fields.values(), 1):
        if not field.types:
            pretty_type = 'N/A'
        elif field.types[0] == GGUFValueType.ARRAY:
            nest_count = len(field.types) - 1
            pretty_type = '[' * nest_count + str(field.types[-1].name) + ']' * nest_count
        else:
            pretty_type = str(field.types[-1].name)
        print(f'  {n:5}: {pretty_type:10} | {len(field.data):8} | {field.name}', end = '')
        if len(field.types) == 1:
            curr_type = field.types[0]
            if curr_type == GGUFValueType.STRING:
                print(' = {0}'.format(repr(str(bytes(field.parts[-1]), encoding='utf8')[:60])), end = '')
            elif field.types[0] in reader.gguf_scalar_to_np:
                print(' = {0}'.format(field.parts[-1][0]), end = '')
        print()

    print(f'\n* Dumping {len(reader.tensors)} tensor(s)')
    for n, tensor in enumerate(reader.tensors, 1):

        prettydims = ', '.join('{0:5}'.format(d) for d in list(tensor.shape) + [1] * (4 - len(tensor.shape)))
        print(f'  {n:5}: {tensor.n_elements:10} | {prettydims} | {tensor.tensor_type.name:7} | {tensor.name}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('dump_gguf: Error: Specify an input file', file = sys.stderr)
        sys.exit(1)
    dump_gguf(sys.argv[1])
