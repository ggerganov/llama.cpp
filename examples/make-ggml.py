#!/usr/bin/env python3
"""
This script converts Hugging Face Llama, StarCoder, Falcon, Baichuan, and GPT-NeoX models to GGUF and quantizes them.

Usage:
python make-ggml.py {model_dir_or_hf_repo_name} --model_type {model_type} [--outname {output_name} (Optional)] [--outdir {output_directory} (Optional)] [--quants {quant_types} (Optional)] [--keep_fp16 (Optional)]

Arguments:
- model: (Required) The directory of the downloaded Hugging Face model or the name of the Hugging Face model repository. If the model directory does not exist, it will be downloaded from the Hugging Face model hub.
- --model_type: (Required) The type of the model to be converted. Choose from llama, starcoder, falcon, baichuan, or gptneox.
- --outname: (Optional) The name of the output model. If not specified, the last part of the model directory path or the Hugging Face model repo name will be used.
- --outdir: (Optional) The directory where the output model(s) will be stored. If not specified, '../models/{outname}' will be used.
- --quants: (Optional) The types of quantization to apply. This should be a space-separated list. The default is 'Q4_K_M Q5_K_S'.
- --keep_fp16: (Optional) If specified, the FP16 model will not be deleted after the quantized models are created.

Old quant types (some base model types require these):
- Q4_0: small, very high quality loss - legacy, prefer using Q3_K_M
- Q4_1: small, substantial quality loss - legacy, prefer using Q3_K_L
- Q5_0: medium, balanced quality - legacy, prefer using Q4_K_M
- Q5_1: medium, low quality loss - legacy, prefer using Q5_K_M

New quant types (recommended):
- Q2_K: smallest, extreme quality loss - not recommended
- Q3_K: alias for Q3_K_M
- Q3_K_S: very small, very high quality loss
- Q3_K_M: very small, very high quality loss
- Q3_K_L: small, substantial quality loss
- Q4_K: alias for Q4_K_M
- Q4_K_S: small, significant quality loss
- Q4_K_M: medium, balanced quality - recommended
- Q5_K: alias for Q5_K_M
- Q5_K_S: large, low quality loss - recommended
- Q5_K_M: large, very low quality loss - recommended
- Q6_K: very large, extremely low quality loss
- Q8_0: very large, extremely low quality loss - not recommended
- F16: extremely large, virtually no quality loss - not recommended
- F32: absolutely huge, lossless - not recommended
"""
import subprocess
subprocess.run(f"pip install huggingface-hub==0.16.4", shell=True, check=True)

import argparse
import os
from huggingface_hub import snapshot_download

def main(model, model_type, outname, outdir, quants, keep_fp16):
    if not os.path.isdir(model):
        print(f"Model not found at {model}. Downloading...")
        try:
            if outname is None:
                outname = model.split('/')[-1]
            model = snapshot_download(repo_id=model, cache_dir='../models/hf_cache')
        except Exception as e:
            raise Exception(f"Could not download the model: {e}")

    if outdir is None:
        outdir = f'../models/{outname}'

    if not os.path.isfile(f"{model}/config.json"):
        raise Exception(f"Could not find config.json in {model}")

    os.makedirs(outdir, exist_ok=True)

    print("Building llama.cpp")
    subprocess.run(f"cd .. && make quantize", shell=True, check=True)

    fp16 = f"{outdir}/{outname}.gguf.fp16.bin"

    print(f"Making unquantised GGUF at {fp16}")
    if not os.path.isfile(fp16):
        if model_type != "llama":
            subprocess.run(f"python3 ../convert-{model_type}-hf-to-gguf.py {model} 1 --outfile {fp16}", shell=True, check=True)
        else:
            subprocess.run(f"python3 ../convert.py {model} --outtype f16 --outfile {fp16}", shell=True, check=True)
    else:
        print(f"Unquantised GGML already exists at: {fp16}")

    print("Making quants")
    for type in quants:
        outfile = f"{outdir}/{outname}.gguf.{type}.bin"
        print(f"Making {type} : {outfile}")
        subprocess.run(f"../quantize {fp16} {outfile} {type}", shell=True, check=True)

    if not keep_fp16:
        os.remove(fp16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert/Quantize HF models to GGUF. If you have the HF model downloaded already, pass the path to the model dir. Otherwise, pass the Hugging Face model repo name. You need to be in the /examples folder for it to work.')
    parser.add_argument('model', help='Downloaded model dir or Hugging Face model repo name')
    parser.add_argument('--model_type', required=True, choices=['llama', 'starcoder', 'falcon', 'baichuan', 'gptneox'], help='Type of the model to be converted. Choose from llama, starcoder, falcon, baichuan, or gptneox.')
    parser.add_argument('--outname', default=None, help='Output model(s) name')
    parser.add_argument('--outdir', default=None, help='Output directory')
    parser.add_argument('--quants', nargs='*', default=["Q4_K_M", "Q5_K_S"], help='Quant types')
    parser.add_argument('--keep_fp16', action='store_true', help='Keep fp16 model', default=False)

    args = parser.parse_args()

    main(args.model, args.model_type, args.outname, args.outdir, args.quants, args.keep_fp16)
