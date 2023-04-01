# Quantizes rwkv.cpp model file from FP32 or FP16 to Q4_0 or Q4_1.
# Usage: python quantize.py bin\Release\rwkv.dll C:\rwkv.cpp-169M-float32.bin C:\rwkv.cpp-169M-q4_1.bin 3

import ctypes
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize rwkv.cpp model file from FP32 or FP16 to Q4_0 or Q4_1')
    parser.add_argument('shared_library_path', help='Path to rwkv.cpp shared library')
    parser.add_argument('src_path', help='Path to FP32/FP16 checkpoint file')
    parser.add_argument('dest_path', help='Path to resulting checkpoint file, will be overwritten')
    parser.add_argument('data_type', help='Data type, 2 (GGML_TYPE_Q4_0) or 3 (GGML_TYPE_Q4_1)', type=int, choices=[2, 3], default=3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    library = ctypes.cdll.LoadLibrary(args.shared_library_path)

    library.rwkv_quantize_model_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    library.rwkv_quantize_model_file.restype = ctypes.c_bool

    result: bool = library.rwkv_quantize_model_file(
        args.src_path.encode('utf-8'),
        args.dest_path.encode('utf-8'),
        ctypes.c_int(args.data_type)
    )

    assert result, 'Failed to quantize, check stderr'

    print('Done')

if __name__ == "__main__":
    main()
