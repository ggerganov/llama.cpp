import os
import struct
import tempfile
import unittest

import numpy as np
from gguf import GGUFWriter


class TestGGUFWriter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _check_file_content(self, file_path, expected_content):
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, "rb") as f:
            file_content = f.read()
            for content in expected_content:
                self.assertTrue(content in file_content)

    def write_and_check_content(self, file_path, gguf_writer, expected_content):
        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        gguf_writer.close()
        self._check_file_content(file_path, expected_content)

    def test_add_uint8(self):
        file_path = os.path.join(self.temp_dir.name, "test0.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.add_uint8("test_uint8", 42)
        self.write_and_check_content(file_path, gguf_writer, [b"test_uint8", b"\x2a"])

    def test_add_int8(self):
        file_path = os.path.join(self.temp_dir.name, "test1.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.add_int8("test_int8", -12)
        self.write_and_check_content(file_path, gguf_writer, [b"test_int8", b"\xf4"])

    def test_add_uint16(self):
        file_path = os.path.join(self.temp_dir.name, "test2.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.add_uint16("test_uint16", 65535)
        self.write_and_check_content(file_path, gguf_writer, [b"test_uint16", b"\xff\xff"])

    def test_write_tensors_to_file(self):
        file_path = os.path.join(self.temp_dir.name, "test3.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.write_header_to_file()
        tensor_data = np.full((10, 10), 1.0, dtype=np.float32)
        gguf_writer.add_tensor("test_tensor", tensor_data)
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()
        expected_content = [b"test_tensor", struct.pack('<f', 1.0)]
        self._check_file_content(file_path, expected_content)

    def test_add_description(self):
        file_path = os.path.join(self.temp_dir.name, "test4.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.add_description("A test model description")
        self.write_and_check_content(file_path, gguf_writer, [b"A test model description"])

    def test_add_uint32(self):
        file_path = os.path.join(self.temp_dir.name, "test5.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        gguf_writer.add_uint32("test_uint32", 4294967295)
        self.write_and_check_content(file_path, gguf_writer, [b"test_uint32", b"\xff\xff\xff\xff"])

    def test_add_float32(self):
        file_path = os.path.join(self.temp_dir.name, "test6.gguf")
        gguf_writer = GGUFWriter(file_path, "gpt2")
        value_to_write = 3.1415926
        gguf_writer.add_float32("test_float32", value_to_write)
        gguf_writer.write_header_to_file()
        gguf_writer.write_kv_data_to_file()
        expected_float_binary = struct.pack('f', value_to_write)
        self.write_and_check_content(file_path, gguf_writer, [b"test_float32", expected_float_binary])


if __name__ == '__main__':
    unittest.main()
