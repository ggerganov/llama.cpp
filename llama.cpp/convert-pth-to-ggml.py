# Compatibility stub

import argparse

import convert

parser = argparse.ArgumentParser(description='Convert a LLaMA model checkpoint to a ggml compatible file')
parser.add_argument('dir_model',  help='directory containing the model checkpoint')
parser.add_argument('ftype',      help='file type (0: float32, 1: float16)', type=int, choices=[0, 1], default=1)
args = parser.parse_args()
convert.main(['--outtype', 'f16' if args.ftype == 1 else 'f32', '--', args.dir_model])
