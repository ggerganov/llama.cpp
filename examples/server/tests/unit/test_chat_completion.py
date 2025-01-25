import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


@pytest.mark.parametrize(
    "model,system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason,jinja,chat_template",
    [
        (None, "Book", "What is the best book", 8, "(Suddenly)+", 77, 8, "length", False, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+", 77, 8, "length", True, None),
        (None, "Book", "What is the best book", 8, "^ blue", 23, 8, "length", True, "This is not a chat template, it is"),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length", False, None),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length", True, None),
    ]
)
def test_chat_completion(model, system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason, jinja, chat_template):
    global server
    server.jinja = jinja
    server.chat_template = chat_template
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    })
    assert res.status_code == 200
    assert "cmpl" in res.body["id"] # make sure the completion id has the expected format
    assert res.body["system_fingerprint"].startswith("b")
    assert res.body["model"] == model if model is not None else server.model_alias
    assert res.body["usage"]["prompt_tokens"] == n_prompt
    assert res.body["usage"]["completion_tokens"] == n_predicted
    choice = res.body["choices"][0]
    assert "assistant" == choice["message"]["role"]
    assert match_regex(re_content, choice["message"]["content"])
    assert choice["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    "system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason",
    [
        ("Book", "What is the best book", 8, "(Suddenly)+", 77, 8, "length"),
        ("You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length"),
    ]
)
def test_chat_completion_stream(system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason):
    global server
    server.model_alias = None # try using DEFAULT_OAICOMPAT_MODEL
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    })
    content = ""
    last_cmpl_id = None
    for data in res:
        choice = data["choices"][0]
        assert data["system_fingerprint"].startswith("b")
        assert "gpt-3.5" in data["model"] # DEFAULT_OAICOMPAT_MODEL, maybe changed in the future
        if last_cmpl_id is None:
            last_cmpl_id = data["id"]
        assert last_cmpl_id == data["id"] # make sure the completion id is the same for all events in the stream
        if choice["finish_reason"] in ["stop", "length"]:
            assert data["usage"]["prompt_tokens"] == n_prompt
            assert data["usage"]["completion_tokens"] == n_predicted
            assert "content" not in choice["delta"]
            assert match_regex(re_content, content)
            assert choice["finish_reason"] == finish_reason
        else:
            assert choice["finish_reason"] is None
            content += choice["delta"]["content"]


def test_chat_completion_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=8,
        seed=42,
        temperature=0.8,
    )
    assert res.system_fingerprint is not None and res.system_fingerprint.startswith("b")
    assert res.choices[0].finish_reason == "length"
    assert res.choices[0].message.content is not None
    assert match_regex("(Suddenly)+", res.choices[0].message.content)


def test_chat_template():
    global server
    server.chat_template = "llama3"
    server.debug = True  # to get the "__verbose" object in the response
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"] == "<s> <|start_header_id|>system<|end_header_id|>\n\nBook<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("response_format,n_predicted,re_content", [
    ({"type": "json_object", "schema": {"const": "42"}}, 6, "\"42\""),
    ({"type": "json_object", "schema": {"items": [{"type": "integer"}]}}, 10, "[ -3000 ]"),
    ({"type": "json_object"}, 10, "(\\{|John)+"),
    ({"type": "sound"}, 0, None),
    # invalid response format (expected to fail)
    ({"type": "json_object", "schema": 123}, 0, None),
    ({"type": "json_object", "schema": {"type": 123}}, 0, None),
    ({"type": "json_object", "schema": {"type": "hiccup"}}, 0, None),
])
def test_completion_with_response_format(response_format: dict, n_predicted: int, re_content: str | None):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "response_format": response_format,
    })
    if re_content is not None:
        assert res.status_code == 200
        choice = res.body["choices"][0]
        assert match_regex(re_content, choice["message"]["content"])
    else:
        assert res.status_code != 200
        assert "error" in res.body


@pytest.mark.parametrize("messages", [
    None,
    "string",
    [123],
    [{}],
    [{"role": 123}],
    [{"role": "system", "content": 123}],
    # [{"content": "hello"}], # TODO: should not be a valid case
    [{"role": "system", "content": "test"}, {}],
])
def test_invalid_chat_completion_req(messages):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages,
    })
    assert res.status_code == 400 or res.status_code == 500
    assert "error" in res.body


def test_chat_completion_with_timings_per_token():
    global server
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "test"}],
        "stream": True,
        "timings_per_token": True,
    })
    for data in res:
        assert "timings" in data
        assert "prompt_per_second" in data["timings"]
        assert "predicted_per_second" in data["timings"]
        assert "predicted_n" in data["timings"]
        assert data["timings"]["predicted_n"] <= 10


TEST_TOOL = {
    "type":"function",
    "function": {
        "name": "test",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "const": True},
            },
            "required": ["success"]
        }
    }
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to run in the ipython interpreter."
                }
            },
            "required": ["code"]
        }
    }
}

CODE_INTEPRETER_TOOL = {
    "type": "code_interpreter",
}


@pytest.mark.parametrize("template_name,n_predict,tool,argument_key", [
    ("meetkai-functionary-medium-v3.1",               128, TEST_TOOL,   "success"),
    ("meetkai-functionary-medium-v3.1",               128, PYTHON_TOOL, "code"),
    ("meetkai-functionary-medium-v3.2",               128, TEST_TOOL,   "success"),
    ("meetkai-functionary-medium-v3.2",               128, PYTHON_TOOL, "code"),
    ("NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use", 128, TEST_TOOL,   "success"),
    ("NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use", 128, PYTHON_TOOL, "code"),
    ("NousResearch-Hermes-3-Llama-3.1-8B-tool_use",   128, TEST_TOOL,   "success"),
    ("NousResearch-Hermes-3-Llama-3.1-8B-tool_use",   128, PYTHON_TOOL, "code"),
    ("meta-llama-Meta-Llama-3.1-8B-Instruct",         128, TEST_TOOL,   "success"),
    ("meta-llama-Meta-Llama-3.1-8B-Instruct",         128, PYTHON_TOOL, "code"),
    ("meta-llama-Llama-3.2-3B-Instruct",              128, TEST_TOOL,   "success"),
    ("meta-llama-Llama-3.2-3B-Instruct",              128, PYTHON_TOOL, "code"),
    ("mistralai-Mistral-Nemo-Instruct-2407",          128, TEST_TOOL,   "success"),
    ("mistralai-Mistral-Nemo-Instruct-2407",          128, PYTHON_TOOL, "code"),
])
def test_completion_with_required_tool(template_name: str, n_predict: int, tool: dict, argument_key: str):
    global server
    # server = ServerPreset.stories15m_moe()
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../tests/chat/templates/{template_name}.jinja'
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "tool_choice": "required",
        "tools": [tool],
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    })
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    assert tool["function"]["name"] == tool_call["function"]["name"]
    actual_arguments = json.loads(tool_call["function"]["arguments"])
    assert argument_key in actual_arguments, f"tool arguments: {json.dumps(actual_arguments)}, expected: {argument_key}"


@pytest.mark.parametrize("template_name,n_predict,tools,tool_choice", [
    ("meetkai-functionary-medium-v3.1",               128, [],            None),
    ("meetkai-functionary-medium-v3.1",               128, [TEST_TOOL],   None),
    ("meetkai-functionary-medium-v3.1",               128, [PYTHON_TOOL], 'none'),
    ("meetkai-functionary-medium-v3.2",               128, [],            None),
    ("meetkai-functionary-medium-v3.2",               128, [TEST_TOOL],   None),
    ("meetkai-functionary-medium-v3.2",               128, [PYTHON_TOOL], 'none'),
    ("meta-llama-Meta-Llama-3.1-8B-Instruct",         128, [],            None),
    ("meta-llama-Meta-Llama-3.1-8B-Instruct",         128, [TEST_TOOL],   None),
    ("meta-llama-Meta-Llama-3.1-8B-Instruct",         128, [PYTHON_TOOL], 'none'),
])
def test_completion_without_tool_call(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None):
    global server
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../tests/chat/templates/{template_name}.jinja'
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "say hello world with python"},
        ],
        "tools": tools if tools else None,
        "tool_choice": tool_choice,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    })
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    assert choice["message"].get("tool_calls") is None, f'Expected no tool call in {choice["message"]}'


@pytest.mark.slow
@pytest.mark.parametrize("tool,expected_arguments,hf_repo,hf_file,template_override", [
    (PYTHON_TOOL,          {"code": "print('Hello, World!')"},   "bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf", None),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello, World!')"},   "bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf", None),
    (PYTHON_TOOL,          {"code": "print('Hello, World!')"},   "bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf", None),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello, World!')"},   "bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf", None),
    (PYTHON_TOOL,          {"code": "print('Hello World!')"},    "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", None),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello World!')"},    "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", None),
    (PYTHON_TOOL,          {"code": "print('Hello, world!')"},   "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF", "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf", ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello, world!')"},   "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF", "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf", ("NousResearch-Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (PYTHON_TOOL,          {"code": "print('Hello World!')"},    "NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf", ("NousResearch-Hermes-3-Llama-3.1-8B", "tool_use")),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello World!')"},    "NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf", ("NousResearch-Hermes-3-Llama-3.1-8B", "tool_use")),
    (PYTHON_TOOL,          {"code": "print(\"Hello, World!\")"}, "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", ("meta-llama-Llama-3.2-3B-Instruct", None)),
    (CODE_INTEPRETER_TOOL, {"code": "print(\"Hello, World!\")"}, "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", ("meta-llama-Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,          {"code": "print("},                   "bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", ("meta-llama-Llama-3.2-3B-Instruct", None)),
    (CODE_INTEPRETER_TOOL, {"code": "print("},                   "bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", ("meta-llama-Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,          {"code": "print(\"hello world\")"},   "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", None),
    (CODE_INTEPRETER_TOOL, {"code": "print("},                   "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", None),
    (PYTHON_TOOL,          {"code": "print('Hello, World!')\n"}, "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q6_K_L.gguf", None),
    (CODE_INTEPRETER_TOOL, {"code": "print('Hello, World!')\n"}, "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q6_K_L.gguf", ("mistralai-Mistral-Nemo-Instruct-2407", None)),
    # TODO: fix this model
    # (PYTHON_TOOL,          {"code": "print('Hello, World!')"},  "bartowski/functionary-small-v3.2-GGUF", "functionary-small-v3.2-Q8_0.gguf", ("meetkai-functionary-medium-v3.2", None)),
    # (CODE_INTEPRETER_TOOL, {"code": "print('Hello, World!')"},  "bartowski/functionary-small-v3.2-GGUF", "functionary-small-v3.2-Q8_0.gguf", ("meetkai-functionary-medium-v3.2", None)),
])
def test_hello_world_tool_call(tool: dict, expected_arguments: dict, hf_repo: str, hf_file: str, template_override: Tuple[str, str | None] | None):
    global server
    server.n_slots = 1
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = 128
    server.model_hf_repo = hf_repo
    server.model_hf_file = hf_file
    if template_override:
        (template_hf_repo, template_variant) = template_override
        server.chat_template_file = f"../../../tests/chat/templates/{template_hf_repo.replace('/', '') + ('-' + template_variant if template_variant else '')}.jinja"
        assert os.path.exists(server.chat_template_file), f"Template file {server.chat_template_file} does not exist. Run `python scripts/get_hf_chat_template.py {template_hf_repo} {template_variant} > {server.chat_template_file}` to download the template."
    server.start(timeout_seconds=15*60)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            # {"role": "user", "content": "say hello world with python"},
            {"role": "user", "content": "Print a hello world message with python"},
        ],
        "tools": [tool],
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.9,
    })
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    if tool["type"] == "function":
        assert tool["function"]["name"] == tool_call["function"]["name"]
    elif tool["type"] == "code_interpreter":
        assert re.match('i?python', tool_call["function"]["name"])
    actual_arguments = json.loads(tool_call["function"]["arguments"])
    assert json.dumps(expected_arguments) == json.dumps(actual_arguments), f"tool arguments: {json.dumps(actual_arguments)}, expected: {json.dumps(expected_arguments)}"


def test_logprobs():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
    )
    output_text = res.choices[0].message.content
    aggregated_text = ''
    assert res.choices[0].logprobs is not None
    assert res.choices[0].logprobs.content is not None
    for token in res.choices[0].logprobs.content:
        aggregated_text += token.token
        assert token.logprob <= 0.0
        assert token.bytes is not None
        assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text


def test_logprobs_stream():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
        stream=True,
    )
    output_text = ''
    aggregated_text = ''
    for data in res:
        choice = data.choices[0]
        if choice.finish_reason is None:
            if choice.delta.content:
                output_text += choice.delta.content
            assert choice.logprobs is not None
            assert choice.logprobs.content is not None
            for token in choice.logprobs.content:
                aggregated_text += token.token
                assert token.logprob <= 0.0
                assert token.bytes is not None
                assert token.top_logprobs is not None
                assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text
