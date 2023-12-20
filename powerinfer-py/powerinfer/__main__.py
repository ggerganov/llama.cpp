
import argparse

from .solver import solve_gpu_split
from .export_split import export_split


if __name__ == "__main__":
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Optimize neuron activation based on VRAM capacity and other parameters.')
    parser.add_argument('--activation', type=str, required=True, help='Path to the directory containing activation data.')
    parser.add_argument('--neuron', type=int, default=8192*4, help='Total number of neurons in the network.')
    parser.add_argument('--capacity', type=int, default=int(8192*4*32*0.1), help='Total VRAM capacity for the model.')
    parser.add_argument('--layer', type=int, default=59, help='Total number of layers in the neural network.')
    parser.add_argument('--vram-capacity', type=int, help='Total VRAM capacity (Bytes) available for splitting')
    parser.add_argument('--batch', type=int, default=256, help='Batch size for processing.')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for splitting a layer across multiple GPUs.')
    parser.add_argument('--output', type=str, required=True, help='File path for the output pickle file.')

    args = parser.parse_args()

    print("solver args:", args)

    solved = solve_gpu_split(
        activation_path=args.activation,
        neuron=args.neuron,
        capacity=args.capacity,
        layer=args.layer,
        batch=args.batch,
        threshold=args.threshold,
    )

    print(f"solved: {solved}, total neurons: {sum(solved)}")

    export_split(
        activations_path=args.activation,
        output_path=args.output,
        solved_list=solved,
        vram_capacity=args.vram_capacity
    )

    print(f"Exported to {args.output}")
