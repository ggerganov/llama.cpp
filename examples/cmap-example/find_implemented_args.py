# search the specified directory for files that include argv[i] == '-f' or '--file' arguments

import os
import re
import collections
import re
import read_common_h


def replace_dashes_with_underscores(filename):
    with open(filename, 'r') as file:
        content = file.read()
        
    # Match '-' surrounded by word characters on both sides and replace with '_'
    replaced_content = re.sub(r'(\w)-(\w)', r'\1_\2', content)
    
    with open(filename, 'w') as file:
        file.write(replaced_content)

def find_arguments(directory):
    arguments = {}

    # Use os.walk() to traverse through files in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    content = file.read()

                    # Search for the expression "params." excluding prefixes and read the attribute without trailing detritus
                    # matches = re.findall(r'(?:^|\s)params\.(.*)(?=[\). <,;}]|\Z)', content)
                    matches = set(re.findall(r'(?:^|\b)params\.([a-zA-Z_0-9]*)(?=[\). <,;}]|\Z)', content))
                    # Remove duplicates from matches list
                    # arguments_list = list(set([match.strip() for match in matches]))

                    # Add the matches to the dictionary
                    arguments[filepath] = matches

    return arguments

def output_results(result):
    sorted_result = collections.OrderedDict(sorted(result.items()))
    all_of_them = set()
    for filename, arguments in sorted_result.items():
        print(f"Filename: \033[32m{filename.split('/')[-1]}\033[0m, arguments: {arguments}\n")
        for argument in arguments:
            if argument not in all_of_them:
                all_of_them.add("".join(argument))
    print(f"\033[32mAll of them: \033[0m{sorted(all_of_them)}.")
    return sorted_result

def concatenate(v):
    concatenated_element = ""
    for i, element in enumerate(v):
        if element == "//":
            concatenated_element = " ".join(v[i:])
    return concatenated_element

def find_parameters(file, sorted_result):
     with open(file, "r") as helpfile:
        lines = helpfile.read().split("\n")
        for filename, arguments in sorted_result.items():
            parameters = []
            for line in lines:
                for argument in arguments:
                    # building pattern to avoid spurious matches
                    pattern = r"(?:--{}\s)|(?:params\.{}[\s.,();])".format(argument, argument.split('n_')[-1])
                    if re.search(pattern, line):
                        parameters.append(line)
    
            all_parameters = set(parameters)
            file = filename.split('/')[-1]
            print("\n\n"+"#"*(10+len(file)))         
            print(f"Filename: \033[32m{file}\033[0m")
            print("#"*(10+len(file))) 
            print(f"\n\n    command-line arguments available and gpt-params functions implemented (TODO: multi-line helps NEED SOME WORK):\n")

            if not all_parameters:
                print(f"    \033[032mNone\033[0m\n")
            
            # first do it the original way 
            else:
                help_count = 0
                for parameter in all_parameters:
                    help_count += 1
                    print(f"{help_count:>2} help: \033[33m{parameter:<30}\033[0m")

                # now do it the new way
                print("\nNow we extract the original gpt_params definition and defaults for implemented arguments:\n")
                gpt_count = 0
                for k,v in read_common_h.parameters.items():
                    if not read_common_h.parameters.items():
                        print(f"    \033[032mNone\033[0m\n")
                    elif k in arguments:
                        # print(f"gpt_params: \033[33m{k:>20}\033[0m values: {v}")
                        concatenated_element = concatenate(v)
                        gpt_count += 1
                        print(f"{gpt_count:>2} gpt_param: \033[32m{k:>19}; \033[34mrole: \033[33m{concatenated_element:<60}\033[0m;  \033[34mdefault: \033[30m{v[1]:<10}\033[0m ")
                
                # searching the other way round is quicker:
                print("\nSearching the other way round is quicker:\n")
                key_count = 0
                for argument in arguments:
                    if argument in read_common_h.parameters:
                        key_count += 1
                        print(f"{key_count:>2} key: {argument:>25}; role: {concatenate(read_common_h.parameters[argument]):<60}; default: {read_common_h.parameters[argument][1]:<10}")
                if help_count == gpt_count and gpt_count == key_count:
                    print("\n\033[032mNo unresolved help-list incompatibilities with this app.\033[0m")
                else:
                    print("\n\033[031mThis app requires some attention regarding help-function consistency.\033[0m")

# Specify the directory you want to search for cpp files
directory = '/Users/edsilm2/llama.cpp/examples'

if __name__ == '__main__':
    # get the parameters from the common.h file utiity we import
    print(read_common_h.parameters)
    # So now we've got the gpt_parameters in this parameters dict

    # First we alter all the hyphenated help words in help-file.txt to underscores
    # replace_dashes_with_underscores('help_list.txt')
    # This above may no longer be needed

    print("\n####################### find parameters #################################")
    # Call the find function to collect all the params.attributes and output the result
    result = find_arguments(directory)

    print("\n######################################## output_results #################################")
    # sort the results and output them
    sorted = output_results(result)

    print("\n######################## find help context parameters #################################")
    # analyse the files and what they contain
    find_parameters("help_list.txt", sorted)