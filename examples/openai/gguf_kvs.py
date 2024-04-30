from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

from gguf.gguf_reader import GGUFReader
from gguf.constants import Keys

class GGUFKeyValues:
    def __init__(self, model: Path):
        self.reader = GGUFReader(model.as_posix())
    def __getitem__(self, key: str):
        if '{arch}' in key:
            key = key.replace('{arch}', self[Keys.General.ARCHITECTURE])
        return self.reader.read_field(self.reader.fields[key])
    def __contains__(self, key: str):
        return key in self.reader.fields
    def keys(self):
        return self.reader.fields.keys()
