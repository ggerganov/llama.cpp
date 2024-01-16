import os
import argparse
import numpy as np

class Node:
    def __init__(self, name, op, backend, shape):
        self.name = name
        self.op = op
        self.backend = backend
        self.shape = shape
        self.prev = []
        self.next = []
        self.in_deg = 0
        self.out_deg = 0

def read_graph(file_path, skip_pattens=[]):
    fin = open(file_path, "r")
    lines = fin.readlines()
    fin.close()
    
    nodes = {}
    edges = []

    def do_skip(name):
        skip = False
        for skip_patten in skip_pattens:
            if skip_patten in name:
                skip = True
                break
        if skip:
            return True
        return False

    
    start = False
    for i, line in enumerate(lines):
        line = line.strip()
        if "Start to print tensors in the computation graph" in line:
            start = True
            continue
        elif "Finish printing tensors in the computation graph" in line:
            start = False
            break
        if start and "Tensor name" in line:
            name = line.split("[")[1].split("]")[0]
            op = lines[i + 1].split("[")[1].split("]")[0]
            backend = lines[i + 2].split("[")[1].split("]")[0]
            shape = lines[i + 3].split("(")[1].split(")")[0]
            shape = list(map(lambda x: int(x), shape.split(", ")))
            node = Node(name, op, backend, shape)
            if do_skip(name):
                continue

            source = lines[i + 4].split("[")[1].split("]")[0]
            source = list(map(lambda x: x, source.split(", ")))
            
            if name.startswith("norm-"):
                ffn_norm = False
                for prev_node in source:
                    if "ffn_inp" in prev_node:
                        ffn_norm = True
                if ffn_norm:
                    name = "ffn_" + name
                else:
                    name = "attn_" + name
            if name in nodes:
                continue
            nodes[name] = node

            for prev_node in source:
                if do_skip(prev_node):
                    continue
                if prev_node not in nodes:
                    nodes[prev_node] = Node(prev_node, "", "", [])
                edges.append((prev_node, name))
                
    for prev, next in edges:
        nodes[next].in_deg += 1
        nodes[next].prev.append(prev)
        nodes[prev].out_deg += 1
        nodes[prev].next.append(next)

    return nodes

def travel_in_topology(nodes, show_path=False):
    degrees = {name: node.in_deg for name, node in nodes.items()}
    concur = 0
    orders = {}
    queue = []

    for name, degree in degrees.items():
        orders[name] = 1
        if degree == 0:
            queue.append(name)

    while len(queue) > 0:
        cur_node = queue.pop(0)
        for next_node in nodes[cur_node].next:
            degrees[next_node] -= 1
            orders[next_node] = np.max((orders[next_node], orders[cur_node] + 1))
            if degrees[next_node] == 0:
                queue.append(next_node)
                if show_path:
                    for prev_node in nodes[next_node].prev:
                        print(f"{prev_node} -> {next_node}")
    
    return concur, orders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True, type=str)
    parser.add_argument("--start_node", type=str, default=None)
    parser.add_argument("--show_path", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    gf = read_graph(args.log_file, skip_pattens=[".weight"])

    concur, orders = travel_in_topology(gf, show_path=args.show_path)
    print(f"max concurrency: {concur}, max order {np.max(list(orders.values()))}")
            