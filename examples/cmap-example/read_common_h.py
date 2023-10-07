# read common.h and extract the parameters name list

import re

# Read the file into separate lines
with open('common/common.h', 'r') as file:
    lines = file.read().split('\n')

parameters = {}
inside = False
for line in lines:
    # non_whitespace_elements = re.findall(r"\S+", line)
    non_whitespace_elements = re.findall(r"[^\s}{=;]+", line)
    print(f"nwe = \033[33m{non_whitespace_elements}\033[0m")
    if non_whitespace_elements and non_whitespace_elements[0] == "struct":
        inside = True
    if len(non_whitespace_elements) > 2 and inside:
        # note: cannot use nwe[0] because types do not generate unique keys and so overwrite
        # here we deliberately add back the key so we can make a manual change when it is different
        parameters[non_whitespace_elements[1]] = non_whitespace_elements[1:]
        for k, v in parameters.items():
            print(f"key: {k:<20}; values: {v}")
            
            concatenated_element = ""
            for i, element in enumerate(v):
                if element == "//":
                    concatenated_element = " ".join(v[i:])
                    # break
            print(" "*10 + f"parameter: \033[32m{k:>40} \033[34mdefault: \033[30m{v[1]:>5} \033[34mcommment: \033[33m{concatenated_element:80}\033[0m")
    
    # this is a bit of a hack to terminate the harvest 
    if len(non_whitespace_elements) > 2 and non_whitespace_elements[1] == "infill":
        inside = False
        break