# read common.h and extract the parameters name list

import re

# Read the file into separate lines
with open('common/common.h', 'r') as file:
    lines = file.read().split('\n')

parameters = {}
# we add the logit_bias parameter which otherwise is not found
parameters['logit_bias']=['logit_bias', '0', '//', 'way', 'to', 'alter', 'prob', 'of', 'particular', 'words']

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
        # remove spurious entry caused by eccentric status of logit_bias
        if "float>" in parameters and parameters["float>"][1] == 'logit_bias':
            del parameters["float>"]

    # this is a bit of a hack to terminate the harvest 
    if len(non_whitespace_elements) > 2 and non_whitespace_elements[1] == "infill":
        inside = False
        break
for k, v in parameters.items():
    print(f"key: {k:<20}; values: {v}")
    concatenated_element = ""
    for i, element in enumerate(v):
        if element == "//":
            concatenated_element = " ".join(v[i:])
            # break
    print(" "*10 + f"parameter: \033[32m{k:>40} \033[34mdefault: \033[30m{v[1]:>5} \033[34mcommment: \033[33m{concatenated_element:80}\033[0m")
