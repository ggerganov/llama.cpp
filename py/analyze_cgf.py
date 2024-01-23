import os
import argparse

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
            nodes[name] = node

            source = lines[i + 4].split("[")[1].split("]")[0]
            source = list(map(lambda x: x, source.split(", ")))
            
            for pre_node in source:
                if do_skip(pre_node):
                    continue
                if pre_node not in nodes:
                    nodes[pre_node] = Node(pre_node, "", "", [])
                edges.append((pre_node, name))
                
    for prev, next in edges:
        nodes[next].in_deg += 1
        nodes[next].prev.append(prev)
        nodes[prev].out_deg += 1
        nodes[prev].next.append(next)

    return nodes

def compute_concur(start, nodes):
    concur = 1
    order = 0
    queue = [(order, start)]
    while len(queue) > 0:
        if order != queue[0][0]:
            concur = len(queue)
            order = queue[0][0]
        cur_order, cur_node = queue.pop(0)
        for next_node in nodes[cur_node].next:
            queue.append((cur_order + 1, next_node))
    return concur

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True, type=str)
    args = parser.parse_args()

    gf = read_graph(args.log_file, skip_pattens=[".weight"])

    max_concur = 1
    for name, node in gf.items():
        if node.in_deg == 0:
            concur = compute_concur(name, gf)
            print(f"Start node: {name}, Max concurrency: {concur}")