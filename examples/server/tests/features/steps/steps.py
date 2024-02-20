import socket
import threading
import time
from contextlib import closing

import openai
import requests
from behave import step


@step(u"a server listening on {server_fqdn}:{server_port} with {n_slots} slots")
def step_server_config(context, server_fqdn, server_port, n_slots):
    context.server_fqdn = server_fqdn
    context.server_port = int(server_port)
    context.n_slots = int(n_slots)
    context.base_url = f'http://{context.server_fqdn}:{context.server_port}'

    context.completions = []
    context.completion_threads = []
    context.prompts = []

    openai.api_key = 'llama.cpp'
    openai.api_base = f'{context.base_url}/v1/chat'


@step(u"the server is {expecting_status}")
def step_wait_for_the_server_to_be_started(context, expecting_status):
    match expecting_status:
        case 'starting':
            server_started = False
            while not server_started:
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    result = sock.connect_ex((context.server_fqdn, context.server_port))
                    if result == 0:
                        return 0
        case 'loading model':
            wait_for_health_status(context, 503, 'loading model')
        case 'healthy':
            wait_for_health_status(context, 200, 'ok')
        case 'ready' | 'idle':
            wait_for_health_status(context, 200, 'ok', params={'fail_on_no_slot': True})
        case 'busy':
            wait_for_health_status(context, 503, 'no slot available', params={'fail_on_no_slot': True})
        case _:
            assert False, "unknown status"


@step(u'a {prompt} completion request with maximum {n_predict} tokens')
def step_request_completion(context, prompt, n_predict):
    request_completion(context, prompt, n_predict)


@step(u'{predicted_n} tokens are predicted')
def step_n_tokens_predicted(context, predicted_n):
    assert_n_tokens_predicted(context.completions[0], int(predicted_n))


@step(u'a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.user_prompt = user_prompt


@step(u'a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@step(u'a model {model}')
def step_model(context, model):
    context.model = model


@step(u'{max_tokens} max tokens to predict')
def step_max_tokens(context, max_tokens):
    context.max_tokens = int(max_tokens)


@step(u'streaming is {enable_streaming}')
def step_streaming(context, enable_streaming):
    context.enable_streaming = enable_streaming == 'enabled' or bool(enable_streaming)


@step(u'an OAI compatible chat completions request')
def step_oai_chat_completions(context):
    oai_chat_completions(context, context.user_prompt)


@step(u'a prompt')
def step_a_prompt(context):
    context.prompts.append(context.text)


@step(u'concurrent completion requests')
def step_concurrent_completion_requests(context):
    concurrent_requests(context, request_completion)


@step(u'concurrent OAI completions requests')
def step_oai_chat_completions(context):
    concurrent_requests(context, oai_chat_completions)


@step(u'all prompts are predicted')
def step_all_prompts_are_predicted(context):
    for completion_thread in context.completion_threads:
        completion_thread.join()
    for completion in context.completions:
        assert_n_tokens_predicted(completion)


def concurrent_requests(context, f_completion):
    context.completions.clear()
    context.completion_threads.clear()
    for prompt in context.prompts:
        completion_thread = threading.Thread(target=f_completion, args=(context, prompt))
        completion_thread.start()
        context.completion_threads.append(completion_thread)
    context.prompts.clear()


def request_completion(context, prompt, n_predict=None):
    response = requests.post(f'{context.base_url}/completion', json={
        "prompt": prompt,
        "n_predict": int(n_predict) if n_predict is not None else 4096,
    })
    status_code = response.status_code
    assert status_code == 200
    context.completions.append(response.json())


def oai_chat_completions(context, user_prompt):
    chat_completion = openai.Completion.create(
        messages=[
            {
                "role": "system",
                "content": context.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model=context.model,
        max_tokens=context.max_tokens,
        stream=context.enable_streaming
    )
    if context.enable_streaming:
        completion_response = {
            'content': '',
            'timings': {
                'predicted_n': 0
            }
        }
        for chunk in chat_completion:
            assert len(chunk.choices) == 1
            delta = chunk.choices[0].delta
            if 'content' in delta:
                completion_response['content'] += delta['content']
                completion_response['timings']['predicted_n'] += 1
        context.completions.append(completion_response)
    else:
        assert len(chat_completion.choices) == 1
        context.completions.append({
            'content': chat_completion.choices[0].message,
            'timings': {
                'predicted_n': chat_completion.usage.completion_tokens
            }
        })


def assert_n_tokens_predicted(completion_response, expected_predicted_n=None):
    content = completion_response['content']
    n_predicted = completion_response['timings']['predicted_n']
    assert len(content) > 0, "no token predicted"
    if expected_predicted_n is not None:
        assert n_predicted == expected_predicted_n, (f'invalid number of tokens predicted:'
                                                     f' "{n_predicted}" <> "{expected_predicted_n}"')


def wait_for_health_status(context, expected_http_status_code, expected_health_status, params=None):
    while True:
        health_response = requests.get(f'{context.base_url}/health', params)
        status_code = health_response.status_code
        health = health_response.json()
        if status_code == expected_http_status_code and health['status'] == expected_health_status:
            break
