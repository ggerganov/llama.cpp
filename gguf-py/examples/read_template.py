#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from gguf.gguf_reader import GGUFReader

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: read_template.py <path_to_gguf_file>")
        sys.exit(1)
    gguf_file_path = sys.argv[1]

    reader = GGUFReader(gguf_file_path)
    print(reader.read_field(reader.fields['tokenizer.chat_template']))
