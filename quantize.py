#!/usr/bin/python3

"""Script to execute quantization on a given model."""

import subprocess
import argparse
import sys
import os


def main():
    """Parse the command line arguments and execute the script."""

    parser = argparse.ArgumentParser(
        prog='Quantization Script',
        description='This script quantizes a model.'
    )

    parser.add_argument("models", nargs='+', dest='models')
    parser.add_argument(
        '-r', '--remove-16', action='store_true', dest='remove_f16',
        help='Remove the f16 model after quantizing it.'
    )

    args = parser.parse_args()

    for model in args.models:

        model_path = os.path.join("models", model, "ggml-model-f16.bin")

        for i in os.listdir(model_path):
            subprocess.run(
                ["./quantize", i, i.replace("f16", "q4_0"), "2"],
                shell=True,
                check=True
            )

            if args.remove_f16:
                os.remove(i)


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError:
        print("An error ocurred while trying to quantize the models.")
        sys.exit(1)

    except FileNotFoundError as err:
        print(
            f'A FileNotFoundError exception was raised while executing the \
            script:\n{err}\nMake sure you are located in the root of the \
            repository and that the models are in the "models" directory.'
        )
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(0)
