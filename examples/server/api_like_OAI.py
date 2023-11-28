#!/usr/bin/env python3
import argparse
from flask import Flask, jsonify, request, Response
import urllib.parse
import requests
import time
import json


app = Flask(__name__)
slot_id = -1

parser = argparse.ArgumentParser(description="An example of using server.cpp with a similar API to OAI. It must be used together with server.cpp.")
parser.add_argument("--chat-prompt", type=str, help="the top prompt in chat completions(default: 'A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.\\n')", default='A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.\\n')
parser.add_argument("--user-name", type=str, help="USER name in chat completions(default: '\\nUSER: ')", default="\\nUSER: ")
parser.add_argument("--ai-name", type=str, help="ASSISTANT name in chat completions(default: '\\nASSISTANT: ')", default="\\nASSISTANT: ")
parser.add_argument("--system-name", type=str, help="SYSTEM name in chat completions(default: '\\nASSISTANT's RULE: ')", default="\\nASSISTANT's RULE: ")
parser.add_argument("--function-name", type=str, help="FUNCTION name in chat completions(default: '\\nFUNCTION: ')", default="\\nFUNCTION: ")
parser.add_argument("--stop", type=str, help="the end of response in chat completions(default: '</s>')", default="</s>")
parser.add_argument("--llama-api", type=str, help="Set the address of server.cpp in llama.cpp(default: http://127.0.0.1:8080)", default='http://127.0.0.1:8080')
parser.add_argument("--api-key", type=str, help="Set the api key to allow only few user(default: NULL)", default="")
parser.add_argument("--host", type=str, help="Set the ip address to listen.(default: 127.0.0.1)", default='127.0.0.1')
parser.add_argument("--port", type=int, help="Set the port to listen.(default: 8081)", default=8081)

args = parser.parse_args()

def is_present(json, key):
    try:
        buf = json[key]
    except TypeError:
        return False
    except KeyError:
        return False
    if json[key] == None:
        return False
    return True

#convert chat to prompt
def convert_chat(messages):
    prompt = "" + args.chat_prompt.replace("\\n", "\n")

    system_n = args.system_name.replace("\\n", "\n")
    user_n = args.user_name.replace("\\n", "\n")
    ai_n = args.ai_name.replace("\\n", "\n")
    fn_n = args.function_name.replace("\\n", "\n")
    stop = args.stop.replace("\\n", "\n")


    for line in messages:
        if (line["role"] == "system"):
            prompt += f"{system_n}{line['content']}"
        if (line["role"] == "user"):
            prompt += f"{user_n}{line['content']}"
        if (line["role"] == "function"):
            prompt += f"{fn_n}{line['content']}"
        if (line["role"] == "assistant"):
            prompt += f"{ai_n}{line['content']}{stop}"
    prompt += ai_n.rstrip()

    return prompt

def make_grammar(schema, root):
    indent_inc = "  "
    def format_rulename(name):
        return name.replace("_", "-")

    def is_basetype(typename):
        return typename in ['integer', 'number', 'bool', 'string']

    def schema_typename(prefix, schema, defs, arrs):
        typename = '"null"'
        if 'type' in schema:
            typename = schema['type']
            if not is_basetype(typename) and typename != 'array':
                typename = prefix + typename
        if '$ref' in schema and schema['$ref'] in defs:
            typename = defs[schema['$ref']]
        if typename == 'array':
            elemtype = schema_typename(prefix, schema['items'], defs, arrs)
            typename = f'array-{elemtype}'
            if not is_basetype(elemtype):
                typename = prefix + typename
            typename = format_rulename(typename)
            if typename not in arrs:
                arrs[typename] = elemtype
        return typename

    def arr_to_rules(rules, prefix, name, elemtype):
        rulename = name
        rulename = format_rulename(rulename)
        rules[rulename] = f'"[" ( {elemtype} ( "," {elemtype} )* )? "]"'

    def enum_to_rules(rules, prefix, name, schema):
        enum_values = schema['enum']
        etype = schema['type']
        def value_pattern(value):
            if etype == 'string':
                return f'"\\"{repr(value)[1:-1]}\\""'
            else:
                return repr(value)
        values = ' | '.join([
            value_pattern(value)
            for value in enum_values
        ])
        rulename = format_rulename(f'{prefix}{name}')
        rules[rulename] = f'( {values} )'

    def anyof_to_rules(rules, prefix, name, schema, defs, arrs):
        values = schema['anyOf']
        values = ' | '.join([
            schema_typename(prefix, value, defs, arrs)
            for value in values
        ])
        rulename = format_rulename(f'{prefix}{name}')
        rules[rulename] = f'( {values} )'

    def declare_rules(indent, rules, prefix, name, schema, defs, arrs):
        if 'enum' in schema:
            enum_to_rules(rules, prefix, format_rulename(name), schema)
        elif 'anyOf' in schema:
            anyof_to_rules(rules, prefix, format_rulename(name), schema, defs, arrs)
        elif schema.get('type', None) == 'object':
            obj_to_rules(indent + indent_inc, rules, prefix, format_rulename(name), schema, defs, arrs, is_toplevel=False)

    def obj_to_rules(indent, rules, prefix, name, schema, defs, arrs, is_toplevel):
        assert(schema['type'] == 'object')
        if defs is None:
            defs = {}
        if arrs is None:
            arrs = {}


        rulename = name
        if not is_toplevel:
            rulename = prefix+rulename
        rulename = format_rulename(rulename)

        if '$defs' in schema:
            for name, _def in schema['$defs'].items():
                defs['#/$defs/' + name] = format_rulename(prefix + name)

        if '$defs' in schema:
            for name, _def in schema['$defs'].items():
                declare_rules(indent + indent_inc, rules, prefix, name, _def, defs, arrs)

        for name, prop in schema['properties'].items():
            declare_rules(indent + indent_inc, rules, prefix, name, prop, defs, arrs)

        def propery_to_grammar(name, typename):
            return f'"\\"" "{name}" "\\"" ":" {typename}'

        properties = ' "," '.join([
            propery_to_grammar(name, schema_typename(prefix, prop, defs, arrs))
            for name, prop in schema['properties'].items()
        ])
        rules[rulename] = f'"{{" {properties} "}}"'

        for arrtype, elemtype in arrs.items():
            arr_to_rules(rules, prefix, arrtype, elemtype)

        return rulename

    def model_grammar(schema, root = None):
        indent = ""
        rules = {}
        defs = {}
        fns = {}
        arrs = {}

        for fn in schema:
            name = fn['name']
            params = fn['parameters']
            prefix = f"{name}-"
            fns[name] = obj_to_rules(indent + indent_inc, rules, prefix, name, params, {}, {}, is_toplevel=True)

        root = format_rulename(fns[root["name"]])

        rules['root'] = root
        for k in rules:
            if callable(rules[k]):
                rules[k] = rules[k]()

        rules = [f'{k} ::= {v}' for k,v in rules.items()]
        grammar = "\n".join(rules)
        grammar += ( # json base types
(r'''
ws ::= [ \t\n]?
string ::=
  "\"" (
    [^"\\] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\""
bool ::= "True" | "False"
integer ::= ("-"? ([0-9] | [1-9] [0-9]*))
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
'''))
        return grammar

    def decision_grammar(schema, root):
        fnames = [fn['name'] for fn in schema]
        fnames = [f'"\\"" "{fn}" "\\""' for fn in fnames]
        fnames = " | ".join(fnames)
        rules = {}
        rules["root"] = 'msg | call'
        rules["msg"]   = '"{" "\\"" "decision" "\\"" ":" "\\"" "message" "\\"" "}"'
        rules["call"]  = '"{" "\\"" "decision" "\\"" ":" "\\"" "function" "\\"" "," "\\"" "function_name" "\\"" ":" fname "}"'
        rules["fname"] = fnames
        rules = [f'{k} ::= {v}' for k,v in rules.items()]
        grammar = "\n".join(rules)
        return grammar

    if root == "auto":
        return decision_grammar(schema, root)
    if root is None:
        return None
    else:
        return model_grammar(schema, root)

def make_postData(body, chat=False, stream=False, decide_function=False, function_call=None):
    postData = {}
    if (chat):
        postData["prompt"] = convert_chat(body["messages"])
    else:
        postData["prompt"] = body["prompt"]

    if(is_present(body, "functions") and len(body["functions"])>0):
        assert(is_present(body, "functions"))
        grammar = make_grammar(body["functions"], function_call)
        if grammar is not None:
            postData["grammar"] = grammar

    if(is_present(body, "temperature")): postData["temperature"] = body["temperature"]
    if(is_present(body, "top_k")): postData["top_k"] = body["top_k"]
    if(is_present(body, "top_p")): postData["top_p"] = body["top_p"]
    if(is_present(body, "max_tokens")): postData["n_predict"] = body["max_tokens"]
    if(is_present(body, "presence_penalty")): postData["presence_penalty"] = body["presence_penalty"]
    if(is_present(body, "frequency_penalty")): postData["frequency_penalty"] = body["frequency_penalty"]
    if(is_present(body, "repeat_penalty")): postData["repeat_penalty"] = body["repeat_penalty"]
    if(is_present(body, "mirostat")): postData["mirostat"] = body["mirostat"]
    if(is_present(body, "mirostat_tau")): postData["mirostat_tau"] = body["mirostat_tau"]
    if(is_present(body, "mirostat_eta")): postData["mirostat_eta"] = body["mirostat_eta"]
    if(is_present(body, "seed")): postData["seed"] = body["seed"]
    if(is_present(body, "logit_bias")): postData["logit_bias"] = [[int(token), body["logit_bias"][token]] for token in body["logit_bias"].keys()]
    if (args.stop != ""):
        postData["stop"] = [args.stop]
    else:
        postData["stop"] = []
    if(is_present(body, "stop")): postData["stop"] += body["stop"]
    postData["n_keep"] = -1
    postData["stream"] = stream
    postData["cache_prompt"] = True
    postData["slot_id"] = slot_id
    return postData

def make_resData(data, chat=False, promptToken=[], function_call={}):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA_CPP",
        "usage": {
            "prompt_tokens": data["tokens_evaluated"],
            "completion_tokens": data["tokens_predicted"],
            "total_tokens": data["tokens_evaluated"] + data["tokens_predicted"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken

    if chat and is_present(function_call, "name"):
            resData["choices"][0]["delta"] = [{
                "index": 0,
                "function_call": {
                    "name": function_call["name"],
                    "arguments": ""
                },
                "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
            }]
            if start:
                resData["choices"][0]["delta"]["role"] = "assistant"
            if is_present(data, "content"):
                resData["choices"][0]["delta"]["function_call"]["arguments"] = data["content"]
    elif (chat):
        #only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    else:
        #only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    return resData

def make_resData_stream(data, chat=False, time_now = 0, start=False, function_call={}):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion.chunk" if (chat) else "text_completion.chunk",
        "created": time_now,
        "model": "LLaMA_CPP",
        "choices": [
            {
                "finish_reason": None,
                "index": 0
            }
        ]
    }
    slot_id = data["slot_id"]
    if (chat):
        if is_present(function_call, "name"):
            resData["choices"][0]["delta"] =  {
                "function_call": {
                    "name": function_call["name"],
                    "arguments" : ""
                }
            }
            if start:
                resData["choices"][0]["delta"]["role"] = "assistant"
            if is_present(data, "content"):
                resData["choices"][0]["delta"]["function_call"]["arguments"] = data["content"]
            if is_present(data, "stop") and data["stop"]:
                resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        elif (start):
            resData["choices"][0]["delta"] =  {
                "role": "assistant"
            }
        else:
            resData["choices"][0]["delta"] =  {
                "content": data["content"]
            }
            if (data["stop"]):
                resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
    else:
        resData["choices"][0]["text"] = data["content"]
        if (data["stop"]):
            resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"

    return resData


@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if (args.api_key != "" and request.headers["Authorization"].split()[1] != args.api_key):
        return Response(status=403)
    body = request.get_json()
    stream = False
    tokenize = False
    function_call = None
    if(is_present(body, "stream")): stream = body["stream"]
    if(is_present(body, "tokenize")): tokenize = body["tokenize"]
    if(is_present(body, "function_call")): function_call = body["function_call"]
    if(is_present(body, "functions") and function_call is None): function_call = "auto"

    if function_call == "auto":
        postDataDecide = make_postData(body, chat=True, stream=False, function_call=function_call)
        dataDecide = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postDataDecide))
        decision = json.loads(dataDecide.content)
        decision = json.loads(decision['content'])
        if decision["decision"] == "message":
            function_call = None
        if decision["decision"] == "function":
            function_call = {"name": decision["function_name"]}

    postData = make_postData(body, chat=True, stream=stream, function_call=function_call)

    promptToken = []
    if (tokenize):
        tokenData = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/tokenize"), data=json.dumps({"content": postData["prompt"]})).json()
        promptToken = tokenData["tokens"]

    if (not stream):
        data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData))
        print(data.json())
        resData = make_resData(data.json(), chat=True, promptToken=promptToken, function_call=function_call)
        return jsonify(resData)
    else:
        def generate():
            data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData), stream=True)
            time_now = int(time.time())
            resData = make_resData_stream({}, chat=True, time_now=time_now, start=True, function_call=function_call)
            yield 'data: {}\n'.format(json.dumps(resData))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    resData = make_resData_stream(json.loads(decoded_line[6:]), chat=True, time_now=time_now, function_call=function_call)
                    yield 'data: {}\n'.format(json.dumps(resData))
        return Response(generate(), mimetype='text/event-stream')


@app.route('/completions', methods=['POST'])
@app.route('/v1/completions', methods=['POST'])
def completion():
    if (args.api_key != "" and request.headers["Authorization"].split()[1] != args.api_key):
        return Response(status=403)
    body = request.get_json()
    stream = False
    tokenize = False
    if(is_present(body, "stream")): stream = body["stream"]
    if(is_present(body, "tokenize")): tokenize = body["tokenize"]
    postData = make_postData(body, chat=False, stream=stream)

    promptToken = []
    if (tokenize):
        tokenData = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/tokenize"), data=json.dumps({"content": postData["prompt"]})).json()
        promptToken = tokenData["tokens"]

    if (not stream):
        data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData))
        print(data.json())
        resData = make_resData(data.json(), chat=False, promptToken=promptToken)
        return jsonify(resData)
    else:
        def generate():
            data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData), stream=True)
            time_now = int(time.time())
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    resData = make_resData_stream(json.loads(decoded_line[6:]), chat=False, time_now=time_now)
                    yield 'data: {}\n'.format(json.dumps(resData))
        return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(args.host, port=args.port)
