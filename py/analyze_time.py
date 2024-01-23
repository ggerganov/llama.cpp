import os
import argparse

def read_data(file_path):
    fin = open(file_path, "r")
    lines = fin.readlines()
    fin.close()

    def split(context):
        data_list = []
        read = False
        data = ""
        for c in context:
            if c == "[":
                read = True
                continue
            elif c == "]":
                read = False
                data_list.append(data)
                data = ""
                continue
            if read:
                data += c
        return data_list

    time_summary = {}
    time_info = {}
    start = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line == "{":
            start = True
            continue
        elif line == "}":
            start = False
            continue
        if start and "ggml_compute_forward" in line:
            data = split(line)
            tensor = data[0]
            op = data[1]
            time_cost = float(data[2])
            if op == "inp_embd":
                layer = 0
            elif "-" in op:
                layer = int(op.split(" ")[0].split("-")[1]) + 1
            else:
                layer = 100
            if layer in time_summary:
                time_info[layer].append({"name": tensor, "op": op, "time": time_cost})
                time_summary[layer] += time_cost
            else:
                time_info[layer] = [{"name": tensor, "op": op, "time": time_cost}]
                time_summary[layer] = time_cost
    
    return time_info, time_summary
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True, type=str)
    args = parser.parse_args()

    time_info, time_summary = read_data(args.log_file)
    for k, v in time_summary.items():
        print(k, v)
