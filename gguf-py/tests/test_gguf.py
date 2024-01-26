import os
import sys
from pathlib import Path
import numpy as np
import unittest

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))
from gguf import GGUFWriter, GGUFReader, GGUFValueType  # noqa: E402

model_file = os.path.join(Path(__file__).parent.parent.parent, "models", "test_writer.gguf")


class TestGGUFReaderWriter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gguf_writer = GGUFWriter(model_file, "llama")

        # gguf_writer.add_architecture()
        gguf_writer.add_block_count(12)
        gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
        gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
        gguf_writer.add_kv("uint8", 1)
        gguf_writer.add_kv("nint8", np.int8(1))
        gguf_writer.add_dict("dict1", {"key1": 2, "key2": "hi", "obj": {"k": 1}})
        gguf_writer.add_array_ex("oArray", [{"k": 4, "o": {"o1": 6}}, {"k": 9}])
        gguf_writer.add_array_ex("cArray", [3, "hi", [1, 2]])
        gguf_writer.add_array_ex("arrayInArray", [[2, 3, 4], [5, 7, 8]])
        gguf_writer.add_kv("tokenizer.ggml.bos_token_id", "bos")
        gguf_writer.add_kv("tokenizer.ggml.add_bos_token", True)
        gguf_writer.add_dict("tokenizer_config", {
                             "/tokenizer.ggml.bos_token_id:bos_token": None, "/tokenizer.ggml.add_bos_token": None})
        gguf_writer.add_array("oldArray", [1, 2, 3])
        gguf_writer.add_custom_alignment(64)

        tensor1 = np.ones((32,), dtype=np.float32) * 100.0
        tensor2 = np.ones((64,), dtype=np.float32) * 101.0
        tensor3 = np.ones((96,), dtype=np.float32) * 102.0

        gguf_writer.add_tensor("tensor1", tensor1)
        gguf_writer.add_tensor("tensor2", tensor2)
        gguf_writer.add_tensor("tensor3", tensor3)

        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file()

        gguf_writer.close()

    def test_rw(self) -> None:
        # test compatibility
        gguf_reader = GGUFReader(model_file)
        self.assertEqual(gguf_reader.alignment, 64)
        v = gguf_reader.get_field("oldArray")
        self.assertIsNotNone(v)
        type, itype = v.getType()
        self.assertEqual(type, GGUFValueType.ARRAY)
        self.assertEqual(itype, GGUFValueType.INT32)
        self.assertListEqual(v.get(), [1,2,3])

    def test_rw_ex(self) -> None:
        gguf_reader = GGUFReader(model_file)
        self.assertEqual(gguf_reader.alignment, 64)

        v = gguf_reader.get_field("uint8")
        self.assertEqual(v.get(), 1)
        self.assertEqual(v.types[0], GGUFValueType.UINT8)
        v = gguf_reader.get_field("nint8")
        self.assertEqual(v.get(), 1)
        self.assertEqual(v.types[0], GGUFValueType.INT8)
        v = gguf_reader.get_field("dict1")
        self.assertIsNotNone(v)
        self.assertListEqual(v.get(), ['key1', 'key2', 'obj'])
        v = gguf_reader.get_field(".dict1.key1")
        self.assertEqual(v.get(), 2)
        v = gguf_reader.get_field(".dict1.key2")
        self.assertEqual(v.get(), "hi")
        v = gguf_reader.get_field(".dict1.obj")
        self.assertListEqual(v.get(), ['k'])
        v = gguf_reader.get_field(".dict1.obj.k")
        self.assertEqual(v.get(), 1)

        v = gguf_reader.get_field("oArray")
        self.assertIsNotNone(v)
        count = v.get()
        self.assertEqual(count, 2)
        type, itype = v.getType()
        self.assertEqual(type, GGUFValueType.ARRAY)
        self.assertEqual(itype, GGUFValueType.OBJ)
        v = gguf_reader.get_field(".oArray[0].k")
        self.assertIsNotNone(v)
        self.assertEqual(v.get(), 4)
        v = gguf_reader.get_field(".oArray[1].k")
        self.assertEqual(v.get(), 9)

        v = gguf_reader.get_field("cArray")
        self.assertIsNotNone(v)
        count = v.get()
        self.assertEqual(count, 3)
        type, itype = v.getType()
        self.assertEqual(type, GGUFValueType.ARRAY)
        self.assertEqual(itype, GGUFValueType.OBJ)
        v = gguf_reader.get_field(".cArray[0]")
        self.assertEqual(v.get(), 3)
        v = gguf_reader.get_field(".cArray[1]")
        self.assertEqual(v.get(), "hi")
        v = gguf_reader.get_field(".cArray[2]")
        self.assertListEqual(v.get(), [1, 2])

        v = gguf_reader.get_field("arrayInArray")
        self.assertIsNotNone(v)
        count = v.get()
        self.assertEqual(count, 2)
        type, itype = v.getType()
        self.assertEqual(type, GGUFValueType.ARRAY)
        self.assertEqual(itype, GGUFValueType.ARRAY)
        v = gguf_reader.get_field(".arrayInArray[0]")
        self.assertListEqual(v.get(), [2, 3, 4])
        v = gguf_reader.get_field(".arrayInArray[1]")
        self.assertListEqual(v.get(), [5, 7, 8])

        v = gguf_reader.get_field("tokenizer.ggml.bos_token_id")
        self.assertEqual(v.get(), "bos")
        v = gguf_reader.get_field("tokenizer.ggml.add_bos_token")
        self.assertEqual(v.get(), True)
        v = gguf_reader.get_field("tokenizer_config")
        self.assertIsNotNone(v)
        self.assertListEqual(v.get(), ["/tokenizer.ggml.bos_token_id:bos_token", "/tokenizer.ggml.add_bos_token"])


if __name__ == '__main__':
    unittest.main()
