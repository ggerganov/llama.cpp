# Quantizes rwkv.cpp model file from FP32 or FP16 to Q4_0 or Q4_1.
# Usage: python quantize.py bin\Release\rwkv.dll C:\rwkv.cpp-169M-float32.bin C:\rwkv.cpp-169M-q4_1.bin 3

import argparse
import rwkv_cpp_shared_library

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize rwkv.cpp model file from FP32 or FP16 to Q4_0 or Q4_1')
    parser.add_argument('src_path', help='Path to FP32/FP16 checkpoint file')
    parser.add_argument('dest_path', help='Path to resulting checkpoint file, will be overwritten')
    parser.add_argument('data_type', help='Data type, 2 (GGML_TYPE_Q4_0) or 3 (GGML_TYPE_Q4_1)', type=int, choices=[2, 3], default=3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    library = rwkv_cpp_shared_library.load_rwkv_shared_library()

    library.rwkv_quantize_model_file(
        args.src_path,
        args.dest_path,
        args.data_type
    )

    print('Done')

if __name__ == "__main__":
    main()
