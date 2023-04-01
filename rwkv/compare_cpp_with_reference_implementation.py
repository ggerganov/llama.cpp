# Compares logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV.
# Usage: python compare_cpp_with_reference_implementation.py C:\RWKV-4-Pile-169M-20220807-8023.pth bin\Release\main_rwkv.exe C:\rwkv.cpp-169M.bin

import argparse
import subprocess
import rwkv_model
import torch
import numpy as np
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description='Compare logits from rwkv.cpp implementation of RWKV with logits from reference implementation of RWKV')
    parser.add_argument('torch_model_path', help='Path to PyTorch checkpoint file')
    parser.add_argument('main_executable_path', help='Path to main rwkv.cpp executable file')
    parser.add_argument('ggml_model_path', help='Path to rwkv.cpp checkpoint file')
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    token_count: int = 64
    # It's not important what exactly these tokens are; just that output of both model matches.
    tokens: List[int] = [(i + 1) for i in range(token_count)]
    state_path: str = './state.bin'
    logits_path: str = './logits.bin'

    reference_model: rwkv_model.RWKV_RNN = rwkv_model.RWKV_RNN(args.torch_model_path)

    ref_logits, ref_state = None, None

    for i in range(token_count):
        token: int = tokens[i]

        print()
        print(f'--- {i + 1}/{token_count} ---')

        subprocess.run(
            [
                args.main_executable_path,
                args.ggml_model_path,
                str(token),
                # If this is the first token, let the script create a new state.
                '' if ref_state is None else state_path,
                state_path,
                logits_path
            ],
            check=True
        )

        with open(logits_path, 'rb') as logits_file:
            actual_logits = torch.tensor(np.frombuffer(logits_file.read(), dtype=np.single))

        ref_logits, ref_state = reference_model.forward(token, ref_state)

        difference: float = (torch.sum(ref_logits - actual_logits) / len(ref_logits)).item()

        print(f'Reference logits: {ref_logits}')
        print(f'Actual logits: {actual_logits}')
        print('Difference per token: %.8f' % (difference,))

        assert abs(difference) <= 0.00005, 'Difference is too big'

    print()
    print('Test passes')

if __name__ == "__main__":
    main()
