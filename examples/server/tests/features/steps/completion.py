from behave import *
import requests


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

