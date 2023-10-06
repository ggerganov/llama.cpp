# search the specified directory for files that include argv[i] == '-f' or '--file' arguments

import os
import re
import collections
import re

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
                    matches = re.findall(r'(?:^|\s)params\.(.*?)(?=[\). <,;}]|\Z)', content)
                    # Remove duplicates from matches list
                    arguments_list = list(set([match.strip() for match in matches]))

                    # Add the matches to the dictionary
                    arguments[filepath] = arguments_list

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

def find_parameters(file, sorted_result):
     with open(file, "r") as helpfile:
        lines = helpfile.read().split("\n")
        for filename, arguments in sorted_result.items():
            parameters = []
            for line in lines:
                for argument in arguments:
                    # need to try to avoid spurious matches
                    argument1 = "--" + argument + " "
                    if argument1 in line:
                        parameters.append(line)
                    # need to try to avoid spurious matches
                    argument2 = "params." + argument.split('n_')[-1]
                    if argument2 in line:
                        parameters.append(line)
                    argument3 = "params." + argument
                    if argument3 in line:
                        parameters.append(line)
            all_parameters = set(parameters)            
            print(f"\n\nFilename: \033[32m{filename.split('/')[-1]}\033[0m\n\n    command-line arguments available and gpt-params functions implemented:\n")
            if not all_parameters:
                print(f"    \033[032mNone\033[0m\n")
            else:
                for parameter in all_parameters:
                    print(f"    help: \033[33m{parameter:<30}\033[0m")


# Specify the directory you want to search for cpp files
directory = '/Users/edsilm2/llama.cpp/examples'

if __name__ == '__main__':
    # First we alter all the hyphenated help words in help-file.txt to underscores
    replace_dashes_with_underscores('help_list.txt')
    # Call the find function and output the result
    result = find_arguments(directory)
    sorted = output_results(result)
    # analyse the files and what they contain
    find_parameters("help_list.txt", sorted)