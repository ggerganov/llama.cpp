import os
import re

def find_arguments(directory):
    arguments = {}

    # Get a list of all .cpp files in the specified directory
    cpp_files = [filename for filename in os.listdir(directory) if filename.endswith('.cpp')]

    # Read each .cpp file and search for the specified expressions
    for filename in cpp_files:
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()

            # Search for the expressions using regular expressions
            matches = re.findall(r'argv\s*\[\s*i\s*\]\s*==\s*([\'"])(?P<arg>-[a-zA-Z]+|\-\-[a-zA-Z]+[a-zA-Z0-9-]*)\1', content)

            # Add the found arguments to the dictionary
            arguments[filename] = [match[1] for match in matches]

    return arguments


# Specify the directory you want to search for cpp files
directory = '/Users/edsilm2/llama.cpp/examples'

# Call the function and print the result
result = find_arguments(directory)
for filename, arguments in result.items():
    print(filename, arguments)