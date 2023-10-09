# search the specified directory for files that include command-line arguments
# these are almost always in the form params.argument; "logit_bias" is one exception
# have yet to investigate fully what "lora_adapter" in server.cpp does since it is not apparently
# accessible from the command-line arg/parameter sequence.

import os
import re
import collections
import re
import read_common_h

# update the source file - usually 'help_list.txt', so the default - in case the source file has been changed
def update_file(file_from, file_to = "help_list.txt"):
    # Open the file_from file
    with open(file_from, "r") as file:
        lines = file.readlines()

    # Find lines starting with "printf(" and ending with ");" (assumes file_from is written in C/C++)
    pattern = r'printf\("\s(.*?)\);'
    matched_lines = [re.search(pattern, line).group(1) for line in lines if re.search(pattern, line)]

    # Save matched lines to file_to
    with open(file_to, "w") as file:
        for line in matched_lines:
            file.write(line + '\n')

# helper fn to make the hyphenated words in a file snake-case for searching
def replace_dashes_with_underscores(filename):
    with open(filename, 'r') as file:
        content = file.read()
        
    # Match '-' surrounded by word characters on both sides and replace with '_'
    replaced_content = re.sub(r'(\w)-(\w)', r'\1_\2', content)
    
    with open(filename, 'w') as file:
        file.write(replaced_content)

# helper fn to make the underscored words in a file hyphenated for print
def replace_underscores_with_dashes(parameter):
    # Match '_' surrounded by word characters on both sides and replace with '-'
    return re.sub(r'(\w)_(\w)', r'\1-\2', parameter)


# find all instances of "params." in the *.cpp files in a directory
def find_arguments(directory):
    arguments = {}

    # Use os.walk() to traverse through files in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.cpp'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as file:
                    content = file.read()

                    # Search for the expression "params." or "params->" excluding prefixes and read the attribute without trailing detritus
                    # matches = re.findall(r'(?:^|\s)params\.(.*)(?=[\). <,;}]|\Z)', content)
                    matches = set(re.findall(r'(?:^|\b)params[->\.]([a-zA-Z_0-9]*)(?=[\). <,;}]|\Z)', content))

                    # Add the matches to the dictionary
                    arguments[filepath] = matches

    return arguments

# output a list of the params.attributes for each file
def output_results(result):
    sorted_result = collections.OrderedDict(sorted(result.items()))
    all_of_them = set()
    for filename, arguments in sorted_result.items():
        arguments.add("help")
        print(f"Filename: \033[32m{filename.split('/')[-1]}\033[0m, arguments: {arguments}\n")
        for argument in arguments:
            if argument not in all_of_them:
                all_of_them.add("".join(argument))
    print(f"\033[32mAll of them: \033[0m{sorted(all_of_them)}.")
    return sorted_result

# put all the words after "//" in a dict back together with spaces
def concatenate(v):
    concatenated_element = ""
    for i, element in enumerate(v):
        if element == "//":
            concatenated_element = " ".join(v[i:])
    return concatenated_element

def title_print(filename):
    title = filename.split('/')[-1]
    print("\n\n"+"#"*(10+len(title)))         
    print(f"Filename: \033[32m{title}\033[0m")
    print("#"*(10+len(title))) 

# list all the equivalences between declarations in common.h and common.cpp that defines the help
# these are used to substitute the searched params.attributes (keys) with help attributes (values)
def substitution_list(parameters):
    # store untrapped parameters as identicals in case we need to change them later
    sub_dict = {"n_threads": "threads",
                "n_ctx": "ctx_size",
                "n_draft" : "draft",
                "n_threads_batch" : "threads_batch",
                "n_chunks" : "chunks",
                "n_batch" : "batch_size",
                "n_sequences" : "sequences",
                "n_parallel" : "parallel",
                "n_beams" : "beams",
                "n_keep" : "keep",
                "n_probs" : "nprobs",
                "path_prompt_cache" : "prompt_cache",
                "input_prefix" : "in_prefix",
                "input_suffix" : "in_suffix",
                "input_prefix_bos" : "in_prefix_bos",
                "antiprompt" : "reverse_prompt",
                "mul_mat_q" : "no_mul_mat_q",
                "use_mmap" : "no_mmap",
                "use_mlock" : "mlock",
                "model_alias" : "alias",
                "tfs_z" : "tfs",
                "use_color" : "color",
                "logit_bias" : "logit_bias",
                "ignore_eos" : "ignore_eos",
                "mirostat_tau" : "mirostat_ent",
                "mirostat_eta" : "mirostat_lr",
                "penalize_nl" : "no_penalize_nl",
                "typical_p" : "typical",
                "mem_size" : "mem_size",
                "mem_buffer" : "mem_buffer",
                "no_alloc" : "no_alloc"
                }
    new_parameters = []
    for parameter in parameters:
        if parameter in sub_dict:
            # we need both for future reference 
            new_parameters.append(parameter)
            new_parameters.append(sub_dict[parameter])
        else:
            new_parameters.append(parameter)
    return new_parameters

# output the lines of the help file
def find_parameters(file, sorted_result):
     with open(file, "r") as helpfile:
        lines = helpfile.read().split("\n")
        for filename, arguments in sorted_result.items():
            # we try to fix up some variant labelling in help_file.txt
            arguments = substitution_list(arguments)
            parameters = []
            for line in lines:
                for argument in arguments:
                    # building pattern to avoid spurious matches
                    # pattern = r"(?:--{}\s)|(?:params\.{}[\s.,\.();])".format(argument, argument.split('n_')[-1])
                    pattern = r"(?:--{}\s)|(?:params\.{}(?=[\s.,\.\(\);]|\.+\w))".format(argument, argument.split('n_')[-1])
                    # pattern = r"(?<=params\.)\w+(?=\.\w+|\.|,|;|\}|\{|\(|\)|\.)"
                    # bit of a hack to exclude --attributes at the end of help comment lines
                    if re.search(pattern, line[:50]):
                        parameters.append(line)
    
            all_parameters = set(parameters)

            title_print(filename)
            print(f"\nCommand-line arguments available and gpt-params functions implemented (TODO: multi-line helps NEED SOME WORK):\n")

            if not all_parameters:
                print(f"    \033[032mNone\033[0m\n")
            
            # first do it the original way 
            else:
                help_count = 0
                for parameter in all_parameters:
                    # reverse the hypthen/underscore pattern just for printing
                    replaced_param = replace_underscores_with_dashes(parameter)
                    if not parameter.startswith("    "):
                        help_count += 1
                        print(f"{help_count:>2} help: \033[33m{replaced_param:<30}\033[0m")
                    else:
                        print(f"   help: \033[33m{replaced_param:<30}\033[0m")

                # now do it the new way
                print("\nNow we extract the original gpt_params definition from common.h with the defaults for implemented arguments:\n")
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
                print("\nSearching the other way round is more efficient:\n")
                key_count = 0
                for argument in set(arguments):
                    if argument in read_common_h.parameters:
                        key_count += 1
                        print(f"{key_count:>2} key: {argument:>25}; role: {concatenate(read_common_h.parameters[argument]):<60}; default: {read_common_h.parameters[argument][1]:<10}")
                if help_count == gpt_count and gpt_count == key_count:
                    print(f"\n\033[032mNo unresolved help-list incompatibilities with \033[33m{filename.split('/')[-1]}\033[0m")
                else:
                    print("\n\033[031mThis app requires some attention regarding help-function consistency.\033[0m")

# Specify the directory you want to search for cpp files
directory = '/Users/edsilm2/llama.cpp/examples'

if __name__ == '__main__':
   
   # update the source help file from C++ source (this works exactly as required)
    update_file("common/common.cpp", "help_list.txt")

    # get the parameters from the common.h file utiity we import
    print(read_common_h.parameters)
    # So now we've got the gpt_parameters in this parameters dict

    # First we alter all the hyphenated help words in help-file.txt to underscores
    # we later reverse these changers before printing the help lines
    replace_dashes_with_underscores('help_list.txt')

    print("\n####################### find parameters #################################")
    # Call the find function to collect all the params.attributes and output the result
    result = find_arguments(directory)

    print("\n######################################## output_results #################################")
    # sort the results and output them
    sorted = output_results(result)

    print("\n######################## find help context parameters #################################")
    # analyse the files and what they contain
    find_parameters("help_list.txt", sorted)