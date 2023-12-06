print("hello llama.cpp, got input:\n" + llm_input  + "\n")

if len(llm_input) > 20:
    llm_output = "Reinterpret with emojis " + llm_input + "?\nSTOP";
else:
    llm_output =  llm_input
