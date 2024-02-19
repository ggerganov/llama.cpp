import openai
import requests
from behave import *

openai.api_key = 'llama.cpp'
openai.api_base = "http://localhost:8080/v1/chat"


@given(u'a prompt {prompt}')
def step_prompt(context, prompt):
    context.prompt = prompt


@when(u'we request a completion')
def step_request_completion(context):
    response = requests.post('http://localhost:8080/completion', json={
        "prompt": context.prompt
    })
    status_code = response.status_code
    assert status_code == 200
    context.response_data = response.json()


@then(u'tokens are predicted')
def step_request_completion(context):
    assert len(context.response_data['content']) > 0
    assert context.response_data['timings']['predicted_n'] > 0


@given(u'a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.user_prompt = user_prompt


@given(u'a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@given(u'a model {model}')
def step_model(context, model):
    context.model = model


@when(u'we request the oai completions endpoint')
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


@then(u'the oai response contains completion tokens')
def step_oai_response_has_completion_tokens(context):
    assert len(context.chat_completion.choices) == 1
    assert len(context.chat_completion.choices[0].message) > 0
    assert context.chat_completion.usage.completion_tokens > 0
