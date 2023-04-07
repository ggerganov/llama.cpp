# Quantizes rwkv.cpp model file from FP32 or FP16 to Q4_0, Q4_1 or Q4_1_O (recommended).
# Usage: python quantize.py bin\Release\rwkv.dll C:\rwkv.cpp-169M-float32.bin C:\rwkv.cpp-169M-q4_1_o.bin 4

import argparse
import rwkv_cpp_shared_library

def parse_args():
    parser = argparse.ArgumentParser(description='Quantize rwkv.cpp model file from FP32 or FP16 to Q4_0 or Q4_1')
    parser.add_argument('src_path', help='Path to FP32/FP16 checkpoint file')
    parser.add_argument('dest_path', help='Path to resulting checkpoint file, will be overwritten')
    parser.add_argument('data_type', help='Data type, 2 (GGML_TYPE_Q4_0), 3 (GGML_TYPE_Q4_1) or 4 (GGML_TYPE_Q4_1_O)', type=int, choices=[2, 3, 4], default=4)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.data_type == 2 or args.data_type == 3:
        print()
        print('WARNING!')
        print('You are using Q4_0 or Q4_1 quantization; it will heavily degrade RWKV quality.')
        print('For best quality preservation, it is recommended to use Q4_1_O.')
        print('More info at https://github.com/saharNooby/rwkv.cpp/issues/12')
        print()

    library = rwkv_cpp_shared_library.load_rwkv_shared_library()

    library.rwkv_quantize_model_file(
        args.src_path,
        args.dest_path,
        args.data_type
    )

    print('Done')

if __name__ == "__main__":
    main()
