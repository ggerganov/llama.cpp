# search the specified directory for files that include argv[i] == '-f' or '--file' arguments

import os
import re

def find_arguments(directory):
    arguments = {}

    # Use os.walk() to traverse through files in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    content = file.read()

                    # Search for the expression "params." and read the attribute without trailing detritus
                    matches = re.findall(r'params\.(.*?)(?=[\). <,;}])', content)

                    # Remove duplicates from matches list
                    arguments_list = list(set([match.strip() for match in matches]))

                    # Add the matches to the dictionary
                    arguments[filepath] = arguments_list

    return arguments


# Specify the directory you want to search for cpp files
directory = '/Users/edsilm2/llama.cpp/examples'

if __name__ == '__main__':
    # Call the find function and print the result
    result = find_arguments(directory)
    all_of_them = set()
    for filename, arguments in result.items():
        print(f"Filename: \033[32m{filename}\033[0m, arguments: {arguments}\n")
        for argument in arguments:
            if argument not in all_of_them:
                all_of_them.add("".join(argument))
    print(f"\033[32mAll of them: \033[0m{sorted(all_of_them)}.")

    with open("help_list.txt", "r") as helpfile:
        lines = helpfile.read().split("\n")
        for filename, arguments in result.items():
            parameters = []
            for line in lines:
                for argument in arguments:
                    if argument in line:
                        parameters.append(line)
            all_parameters = set(parameters)            
            print(f"\n\nFilename: \033[32m{filename.split('/')[-1]}\033[0m\n\n    command-line arguments available and gpt-params functions implemented:\n")
            if not all_parameters:
                print(f"    \033[032mNone\033[0m\n")
            else:
                for parameter in all_parameters:
                    print(f"    help: \033[33m{parameter:<30}\033[0m")