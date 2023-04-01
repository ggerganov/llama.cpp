# Compares logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV.
# Reference logits were generated with RWKV-4-Pile-169M-20220807-8023.pth model in PyTorch.
# Reference implementation code: https://github.com/BlinkDL/ChatRWKV/blob/0d0abf181356c6f27501274cad18bdf28c83a45b/RWKV_in_150_lines.py
# Usage: python compare_cpp_with_reference_implementation.py bin\Release\main_rwkv.exe C:\rwkv.cpp-169M.bin

import os
import argparse
import subprocess
import torch
import numpy as np
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description='Compare logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV')
    parser.add_argument('main_executable_path', help='Path to main rwkv.cpp executable file')
    parser.add_argument('ggml_model_path', help='Path to rwkv.cpp checkpoint file')
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Don't want to depend on tokenizer here.
    # Exact string is:
    # context = "1 In the beginning God created the heaven and the earth. " \
    #           "2 And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. " \
    #           "3 And God said, Let there be light: and there was light. " \
    #           "4 And God saw the light, that it was good: and God divided the light from the darkness."
    # The Bible was the first non-copyrighted public domain text that came to my mind.
    tokens: List[int] = [18, 496, 253, 5068, 2656, 3562, 253, 13926, 285, 253, 6149, 15, 374, 1244, 253, 6149, 369, 1293, 830,
                         13, 285, 2991, 28, 285, 13862, 369, 2220, 253, 2454, 273, 253, 3676, 15, 1244, 253, 14559, 273, 2656,
                         4395, 2220, 253, 2454, 273, 253, 12685, 15, 495, 1244, 2656, 753, 13, 1281, 627, 320, 1708, 27, 285,
                         627, 369, 1708, 15, 577, 1244, 2656, 3047, 253, 1708, 13, 326, 352, 369, 1175, 27, 285, 2656, 4272,
                         253, 1708, 432, 253, 13862, 15]

    def compare_logits(tokens_subset: List[int]) -> None:
        token_count: int = len(tokens_subset)
        state_path: str = './state.bin'
        logits_path: str = './logits.bin'

        for i in range(token_count):
            token: int = tokens_subset[i]

            print(f'{i + 1}/{token_count}')

            subprocess.run(
                [
                    args.main_executable_path,
                    args.ggml_model_path,
                    str(token),
                    # If this is the first token, let the script create a new state.
                    '' if i == 0 else state_path,
                    state_path,
                    logits_path
                ],
                check=True
            )

        expected_logits_path: str = f'expected_logits_169M_20220807_8023_{len(tokens_subset)}_tokens.bin'

        if not os.path.isfile(expected_logits_path):
            expected_logits_path = 'rwkv/' + expected_logits_path

        with open(expected_logits_path, 'rb') as logits_file:
            expected_logits = torch.tensor(np.frombuffer(logits_file.read(), dtype=np.single))

        with open(logits_path, 'rb') as logits_file:
            actual_logits = torch.tensor(np.frombuffer(logits_file.read(), dtype=np.single))

        difference: float = (torch.sum(expected_logits - actual_logits) / len(expected_logits)).item()

        print(f'Reference logits: {expected_logits}')
        print(f'Actual logits: {actual_logits}')
        print('Difference per token: %.8f' % (difference,))

        assert abs(difference) <= 0.00005, 'Difference is too big'

    # Check small token amount first to avoid waiting too long before seeing that model is broken
    compare_logits(tokens[:4])
    compare_logits(tokens)

    print()
    print('Test passes')

if __name__ == "__main__":
    main()
