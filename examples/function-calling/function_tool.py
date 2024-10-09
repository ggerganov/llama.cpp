# Generate function calling definitions function schemas

import inspect
import re

# Extract OpenAI function calling style definitions from functions
#
# Generated with: Create a python function to to generate the OpenAI function calling definition from a given function, getting the description, parameter type and parameter description from the function documentation, assuming the function documentation contains sphynx style parameter descriptions, marked with :param.
def get_function_tool_json(func):
    typemap = { 'str': 'string' };
    def get_type(s):
        return typemap[s] if s in typemap else s

    function_name = func.__name__
    doc_parts = re.split(r'\n\s*:param[^:]*\s+', func.__doc__.rstrip());

    function_description = doc_parts[0]
    params_doc = [ re.split(r'\:\s*', param_doc, maxsplit=1) for param_doc in doc_parts[1:] ]
    params_doc = { param: desc for param, desc in params_doc }

    function_def = {
        'name': function_name,
        'description': function_description,
        'parameters': { 'type': 'object', 'properties': {}, 'required': [] }
    }

    for param_name, param in inspect.signature(func).parameters.items():
        function_def['parameters']['properties'][param_name] = {
            'type' : get_type(param.annotation.__name__) if param.annotation is not param.empty else '',
            'description': params_doc[param_name] if param_name in params_doc else ''
        }
        function_def['parameters']['required'].append(param_name);

    return function_def

# Generate function definition schema from function definitions
#
# This is from llama-cpp-python, llama_chat_format.py
def generate_schema_from_functions(functions, namespace="functions") -> str:
    schema = (
        "// Supported function definitions that should be called when necessary.\n"
    )
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        function_name = function["name"]
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        required_params = parameters.get("required", [])

        schema += f"// {description}\n"
        schema += f"type {function_name} = (_: {{\n"

        for param_name, param in parameters.get("properties", {}).items():
            param_description = param.get("description", "")
            param_type = param.get("type", "any")
            optional_indicator = "" if param_name in required_params else "?"
            schema += f"// {param_description}\n"
            schema += f"{param_name}{optional_indicator}: {param_type},\n"
        schema += "}) => any;\n\n"

    schema += "}} // namespace {}".format(namespace)
    return schema
