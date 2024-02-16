# Adjust the main.cpp file
# to include the (Tokens used:) data output

try:
    with open("/Users/edsilm2/llama.cpp/examples/main/main.cpp", 'r+') as file:
        main = file.read()
        search_str = 'printf("\\n> ");'
        new_str = 'printf("\\033[31m(Tokens used: %d / %d)\\033[0m\\nJCP: ", n_past, n_ctx);'
        main = main.replace(search_str, new_str)
        file.seek(0)
        search_str = 'context full and n_predict == -%d => stopping'
        new_str = 'context full and n_predict == %d => stopping'
        main = main.replace(search_str, new_str)
        file.seek(0)
        file.write(main)
except FileNotFoundError as fe:
    print(f"Error searching for main.cpp: {fe}")
