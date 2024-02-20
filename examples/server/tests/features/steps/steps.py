import socket
import threading
import time
from contextlib import closing

import openai
import requests
from behave import step
from behave.api.async_step import async_run_until_complete

base_fqdn = 'localhost'
base_port = 8080
base_url = f"http://{base_fqdn}:{base_port}"

openai.api_key = 'llama.cpp'
openai.api_base = f"{base_url}/v1/chat"

slow_prompt = 'say hello ' * 10
fast_prompt = 'Write a joke'

n_slots = 2


@step(u'wait for the server to be started')
def step_wait_for_the_server_to_be_started(context):
    server_started = False
    while not server_started:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            result = sock.connect_ex((base_fqdn, base_port))
            if result != 0:
                print("server not ready: ", base_fqdn, base_port, result)
                time.sleep(1)
            else:
                return 0


@step(u'wait for the server to be healthy')
def step_wait_for_the_server_to_be_healthy(context):
    status_code = 500
    while status_code != 200:
        status_code = requests.get(f'{base_url}/health').status_code
        if status_code != 200:
            time.sleep(1)


@step(u'an health liveness probe')
def step_an_health_liveness_probe(context):
    response = requests.get(f'{base_url}/health')
    context.status_code = response.status_code
    context.response_data = response.json()


@step(u'the server must be healthy')
def step_server_healthy(context):
    assert context.status_code == 200
    assert context.response_data['status'] == 'ok'


@step(u'the server is overloaded')
@async_run_until_complete()
async def step_server_overloaded(context):
    response = requests.get(f'{base_url}/health?fail_on_no_slot')
    assert response.status_code == 503
    assert response.json()['status'] == 'no slot available'


@step(u'a prompt {prompt}')
def step_prompt(context, prompt):
    context.prompt = prompt


@step(u'we request a completion')
def step_request_completion(context):
    response = requests.post(f'{base_url}/completion', json={
        "prompt": context.prompt
    })
    status_code = response.status_code
    assert status_code == 200
    context.response_data = response.json()


@step(u'tokens are predicted')
def step_request_completion(context):
    prompt_predicted(context.response_data)


@step(u'a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.user_prompt = user_prompt


@step(u'a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@step(u'a model {model}')
def step_model(context, model):
    context.model = model


@step(u'we request the oai completions endpoint')
def step_oai_completions(context):
    context.chat_completion = openai.Completion.create(
        messages=[
            {
                "role": "system",
                "content": context.system_prompt,
            },
            {
                "role": "user",
                "content": context.user_prompt,
            }
        ],
        model=context.model,
    )


@step(u'the oai response contains completion tokens')
def step_oai_response_has_completion_tokens(context):
    assert len(context.chat_completion.choices) == 1
    assert len(context.chat_completion.choices[0].message) > 0
    assert context.chat_completion.usage.completion_tokens > 0


def async_prompt(context, prompt):
    response = requests.post(f'{base_url}/completion', json={
        "prompt": prompt
    })

    context.async_responses.append(response)


@step(u'{n_prompt} {prompt_type} concurrent prompts')
def step_n_concurrent_prompts(context, n_prompt, prompt_type):
    prompt = fast_prompt
    if prompt_type == 'slow':
        prompt = slow_prompt
    context.async_responses = []
    context.threads = []
    for i in range(int(n_prompt)):
        thread = threading.Thread(target=async_prompt, args=(context, prompt))
        thread.start()
        context.threads.append(thread)


def wait_for_slots_processing(context, expected_slots_processing):
    while True:
        health = requests.get(f'{base_url}/health').json()
        if 'slots_processing' in health:  # FIXME when #5594 is merged
            slots_processing = health['slots_processing']
        else:
            slots_processing = 0
        if slots_processing == expected_slots_processing:
            break
        else:
            time.sleep(0.2)


@step(u'wait for all slots processing')
def step_wait_for_all_slots_processing(context):
    wait_for_slots_processing(context, n_slots)


@step(u'wait for all slots idle')
def step_wait_for_all_slots_idle(context):
    wait_for_slots_processing(context, 0)


@step(u'all prompts must be predicted')
def step_all_prompts_must_be_predicted(context):
    for thread in context.threads:
        thread.join()
    for async_response in context.async_responses:
        assert async_response.status_code == 200
        response_data = async_response.json()
        prompt_predicted(response_data)


def prompt_predicted(response_data):
    assert len(response_data['content']) > 0
    assert response_data['timings']['predicted_n'] > 0
