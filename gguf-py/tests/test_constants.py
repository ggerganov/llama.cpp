import sys
from pathlib import Path
import numpy as np
import unittest

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.constants import GGUFValueType  # noqa: E402


class TestGGUFValueType(unittest.TestCase):

    def test_get_type(self):
        self.assertEqual(GGUFValueType.get_type("test"), GGUFValueType.STRING)
        self.assertEqual(GGUFValueType.get_type([1, 2, 3]), GGUFValueType.ARRAY)
        self.assertEqual(GGUFValueType.get_type(1.0), GGUFValueType.FLOAT32)
        self.assertEqual(GGUFValueType.get_type(True), GGUFValueType.BOOL)
        self.assertEqual(GGUFValueType.get_type(b"test"), GGUFValueType.STRING)
        self.assertEqual(GGUFValueType.get_type(np.uint8(1)), GGUFValueType.UINT8)
        self.assertEqual(GGUFValueType.get_type(np.uint16(1)), GGUFValueType.UINT16)
        self.assertEqual(GGUFValueType.get_type(np.uint32(1)), GGUFValueType.UINT32)
        self.assertEqual(GGUFValueType.get_type(np.uint64(1)), GGUFValueType.UINT64)
        self.assertEqual(GGUFValueType.get_type(np.int8(-1)), GGUFValueType.INT8)
        self.assertEqual(GGUFValueType.get_type(np.int16(-1)), GGUFValueType.INT16)
        self.assertEqual(GGUFValueType.get_type(np.int32(-1)), GGUFValueType.INT32)
        self.assertEqual(GGUFValueType.get_type(np.int64(-1)), GGUFValueType.INT64)
        self.assertEqual(GGUFValueType.get_type(np.float32(1.0)), GGUFValueType.FLOAT32)
        self.assertEqual(GGUFValueType.get_type(np.float64(1.0)), GGUFValueType.FLOAT64)
        self.assertEqual(GGUFValueType.get_type({"k": 12}), GGUFValueType.OBJ)


if __name__ == '__main__':
    unittest.main()
