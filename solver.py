#!/usr/bin/env python
# coding=utf-8
import argparse
from cvxopt.glpk import ilp
import numpy as np
from cvxopt import matrix
import torch
import pickle

# Set up command line arguments
parser = argparse.ArgumentParser(description='Optimize neuron activation based on VRAM capacity and other parameters.')
parser.add_argument('--activation_path', type=str, required=True, help='Path to the directory containing activation data.')
parser.add_argument('--neuron', type=int, default=8192*4, help='Total number of neurons in the network.')
parser.add_argument('--capacity', type=int, default=int(8192*4*32*0.1), help='Total VRAM capacity for the model.')
parser.add_argument('--layer', type=int, default=59, help='Total number of layers in the neural network.')
parser.add_argument('--batch', type=int, default=32, help='Batch size for processing.')
parser.add_argument('--threshold', type=int, default=512, help='Threshold for splitting a layer across multiple GPUs.')
parser.add_argument('--output', type=str, required=True, help='File path for the output pickle file.')

args = parser.parse_args()

# Assigning command line arguments to variables
activation_path = args.activation_path
neuron = args.neuron
layer = args.layer
batch = args.batch
output_path = args.output

# Processing activation data
values = []
for i in range(layer):
    # Load and sort activation data for each layer
    freq = torch.load(f"{activation_path}/activation_{i}.pt")
    freq, _ = torch.sort(freq, descending=True)
    freq = freq * -1.0
    freq = freq.view(-1, batch)
    freq = freq.sum(dim=1)
    freq = freq.tolist()
    values += freq

# Padding zero values for additional constraints
for i in range(layer):
    values += [0.0]
c = np.array(values, dtype=float)
c = matrix(c)

# Setting capacity and neuron count per batch
CAP = args.capacity
CAP = int(CAP / batch)
neuron = int(neuron / batch)
coeff = []
h = []

# Constraint 1: Total neuron activation constraint
lst = []
for i in range(neuron * layer):
    lst.append(1)
for i in range(layer):
    lst.append(0)
coeff.append(lst)
h.append(CAP)

# Constraint 2: Threshold constraint for GPU split per layer
for i in range(layer):
    lst = [0] * (neuron * layer + layer)
    for j in range(neuron):
        lst[i * neuron + j] = -1
    lst[neuron * layer + i] = int(args.threshold / batch)
    coeff.append(lst)
    h.append(0)

# Constraint 3: Upper bound on neuron activations
for i in range(layer):
    lst = [0] * (neuron * layer + layer)
    for j in range(neuron):
        lst[i * neuron + j] = 1
    lst[neuron * layer + i] = -1000000  # Arbitrary large negative number as an upper bound
    coeff.append(lst)
    h.append(0)

# Convert lists to matrix format for ILP solver
coeff = np.array(coeff, dtype=float)
G = matrix(coeff)
h = np.array(h, dtype=float)
h = matrix(h)

# Define the set of integer and binary variables
I = set(range(neuron * layer + layer))
B = set()

# Solving the ILP problem
(status, x) = ilp(c, G, h, None, None, B, I)
print(f"ILP Status: {status}")
ans = list(x)
print(f"Total Activation Units: {sum(ans)}")

# Serialize the solution
serialize = []
for i in range(layer):
    serialize.append(sum(ans[i * neuron:i * neuron + neuron] * batch))

aligned_lst = serialize

# Save the solution to a pickle file
with open(output_path, 'wb') as handle:
    pickle.dump(aligned_lst, handle)
