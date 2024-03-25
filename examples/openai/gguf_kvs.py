from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

from gguf.gguf_reader import GGUFReader
from gguf.constants import Keys

class GGUFKeyValues:
    def __init__(self, model: Path):
        reader = GGUFReader(model.as_posix())
        self.fields = reader.fields
    def __getitem__(self, key: str):
        if '{arch}' in key:
            key = key.replace('{arch}', self[Keys.General.ARCHITECTURE])
        return self.fields[key].read()
    def __contains__(self, key: str):
        return key in self.fields
    def keys(self):
        return self.fields.keys()
