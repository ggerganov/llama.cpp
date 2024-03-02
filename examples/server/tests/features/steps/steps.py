import asyncio
import collections
import json
import os
import re
import socket
import subprocess
import time
from contextlib import closing
from re import RegexFlag

import aiohttp
import openai
from behave import step
from behave.api.async_step import async_run_until_complete
from huggingface_hub import hf_hub_download
from prometheus_client import parser


@step(u"a server listening on {server_fqdn}:{server_port}")
def step_server_config(context, server_fqdn, server_port):
    context.server_fqdn = server_fqdn
    context.server_port = int(server_port)
    if 'PORT' in os.environ:
        context.server_port = int(os.environ['PORT'])
        print(f"$PORT set, overriding server port with to {context.server_port}")

    context.base_url = f'http://{context.server_fqdn}:{context.server_port}'

    context.model_alias = None
    context.n_batch = None
    context.n_ctx = None
    context.n_ga = None
    context.n_ga_w = None
    context.n_gpu_layer = None
    context.n_predict = None
    context.n_server_predict = None
    context.n_slots = None
    context.prompt_prefix = None
    context.prompt_suffix = None
    context.server_api_key = None
    context.server_continuous_batching = False
    context.server_embeddings = False
    context.server_metrics = False
    context.server_process = None
    context.seed = None
    context.server_seed = None
    context.user_api_key = None

    context.tasks_result = []
    context.concurrent_tasks = []
    context.prompts = []


@step(u'a model file {hf_file} from HF repo {hf_repo}')
def step_download_hf_model(context, hf_file, hf_repo):
    context.model_file = hf_hub_download(repo_id=hf_repo, filename=hf_file)
    if context.debug:
        print(f"model file: {context.model_file}\n")


@step(u'a model alias {model_alias}')
def step_model_alias(context, model_alias):
    context.model_alias = model_alias


@step(u'{seed:d} as server seed')
def step_seed(context, seed):
    context.server_seed = seed


@step(u'{ngl:d} GPU offloaded layers')
def step_n_gpu_layer(context, ngl):
    if 'N_GPU_LAYERS' in os.environ:
        new_ngl = int(os.environ['N_GPU_LAYERS'])
        if context.debug:
            print(f"-ngl upgraded from {ngl} to {new_ngl}")
        ngl = new_ngl
    context.n_gpu_layer = ngl


@step(u'{n_ctx:d} KV cache size')
def step_n_ctx(context, n_ctx):
    context.n_ctx = n_ctx


@step(u'{n_slots:d} slots')
def step_n_slots(context, n_slots):
    context.n_slots = n_slots


@step(u'{n_predict:d} server max tokens to predict')
def step_server_n_predict(context, n_predict):
    context.n_server_predict = n_predict


@step(u'continuous batching')
def step_server_continuous_batching(context):
    context.server_continuous_batching = True


@step(u'embeddings extraction')
def step_server_embeddings(context):
    context.server_embeddings = True


@step(u'prometheus compatible metrics exposed')
def step_server_metrics(context):
    context.server_metrics = True


@step(u"the server is starting")
def step_start_server(context):
    start_server_background(context)
    attempts = 0
    while True:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            result = sock.connect_ex((context.server_fqdn, context.server_port))
            if result == 0:
                print("\x1b[33;46mserver started!\x1b[0m")
                return
            attempts += 1
            if attempts > 20:
                assert False, "server not started"
            print(f"waiting for server to start, connect error code = {result}...")
            time.sleep(0.1)


@step(u"the server is {expecting_status}")
@async_run_until_complete
async def step_wait_for_the_server_to_be_started(context, expecting_status):
    match expecting_status:
        case 'healthy':
            await wait_for_health_status(context, context.base_url, 200, 'ok')

        case 'ready' | 'idle':
            await wait_for_health_status(context, context.base_url, 200, 'ok',
                                         timeout=10,
                                         params={'fail_on_no_slot': 0, 'include_slots': 0},
                                         slots_idle=context.n_slots,
                                         slots_processing=0,
                                         expected_slots=[{'id': slot_id, 'state': 0}
                                                         for slot_id in
                                                         range(context.n_slots if context.n_slots else 1)])
        case 'busy':
            await wait_for_health_status(context, context.base_url, 503,
                                         'no slot available',
                                         params={'fail_on_no_slot': 0, 'include_slots': 0},
                                         slots_idle=0,
                                         slots_processing=context.n_slots,
                                         expected_slots=[{'id': slot_id, 'state': 1}
                                                         for slot_id in
                                                         range(context.n_slots if context.n_slots else 1)])
        case _:
            assert False, "unknown status"


@step(u'all slots are {expected_slot_status_string}')
@async_run_until_complete
async def step_all_slots_status(context, expected_slot_status_string):
    match expected_slot_status_string:
        case 'idle':
            expected_slot_status = 0
        case 'busy':
            expected_slot_status = 1
        case _:
            assert False, "unknown status"

    expected_slots = [{'id': slot_id, 'state': expected_slot_status}
                      for slot_id in range(context.n_slots)]
    await request_slots_status(context, expected_slots)


@step(u'a completion request with {api_error} api error')
@async_run_until_complete
async def step_request_completion(context, api_error):
    expect_api_error = api_error == 'raised'
    completion = await request_completion(context.prompts.pop(),
                                          context.base_url,
                                          debug=context.debug,
                                          n_predict=context.n_predict,
                                          seed=await completions_seed(context),
                                          expect_api_error=expect_api_error,
                                          user_api_key=context.user_api_key)
    context.tasks_result.append(completion)
    if context.debug:
        print(f"Completion response: {completion}\n")
    if expect_api_error:
        assert completion == 401, f"completion must be an 401 status code: {completion}"


@step(u'{predicted_n:d} tokens are predicted matching {re_content}')
def step_n_tokens_predicted_with_content(context, predicted_n, re_content):
    assert_n_tokens_predicted(context.tasks_result.pop(), predicted_n, re_content)


@step(u'{predicted_n:d} tokens are predicted')
def step_n_tokens_predicted(context, predicted_n):
    assert_n_tokens_predicted(context.tasks_result.pop(), predicted_n)


@step(u'a user prompt {user_prompt}')
def step_user_prompt(context, user_prompt):
    context.prompts.append(user_prompt)


@step(u'a system prompt {system_prompt}')
def step_system_prompt(context, system_prompt):
    context.system_prompt = system_prompt


@step(u'a model {model}')
def step_model(context, model):
    context.model = model


@step(u'{max_tokens:d} max tokens to predict')
def step_max_tokens(context, max_tokens):
    context.n_predict = max_tokens


@step(u'streaming is {enable_streaming}')
def step_streaming(context, enable_streaming):
    context.enable_streaming = enable_streaming == 'enabled'


@step(u'a user api key {user_api_key}')
def step_user_api_key(context, user_api_key):
    context.user_api_key = user_api_key


@step(u'no user api key')
def step_no_user_api_key(context):
    context.user_api_key = None


@step(u'a user api key ')
def step_no_user_api_key_space(context):
    context.user_api_key = None


@step(u'a server api key {server_api_key}')
def step_server_api_key(context, server_api_key):
    context.server_api_key = server_api_key


@step(u'{n_junk:d} as number of junk')
def step_n_junk(context, n_junk):
    context.n_junk = n_junk


@step(u'{n_batch:d} as batch size')
def step_n_batch(context, n_batch):
    context.n_batch = n_batch


@step(u'{seed:d} as seed')
def step_seed(context, seed):
    context.seed = seed


@step(u'a prefix prompt')
def step_prompt_prefix(context):
    context.prompt_prefix = context.text


@step(u'a junk suffix prompt')
def step_prompt_junk_suffix(context):
    context.prompt_junk_suffix = context.text


@step(u'a suffix prompt')
def step_prompt_suffix(context):
    context.prompt_suffix = context.text


@step(u'{n_ga:d} group attention factor'
      u' to extend context size through self-extend')
def step_impl(context, n_ga):
    context.n_ga = n_ga


@step(u'{n_ga_w:d} group attention width to extend context size through self-extend')
def step_impl(context, n_ga_w):
    context.n_ga_w = n_ga_w


@step(u'a passkey prompt template')
def step_prompt_passkey(context):
    context.prompt_passkey = context.text


@step(u'a "{passkey}" passkey challenge prompt with the passkey inserted every {i_pos:d} junk')
def step_prompt_passkey(context, passkey, i_pos):
    prompt = ""
    for i in range(context.n_junk):
        if i % context.n_junk == i_pos:
            prompt += context.prompt_passkey # the passkey is already substituted
        prompt += context.prompt_junk_suffix
    if context.debug:
        passkey_highlight = "\x1b[33m" + passkey + "\x1b[0m"
        print(f"Passkey challenge:\n```{prompt.replace(passkey, passkey_highlight)}```\n")
    context.prompts.append(context.prompt_prefix + prompt + context.prompt_suffix)


@step(u'an OAI compatible chat completions request with {api_error} api error')
@async_run_until_complete
async def step_oai_chat_completions(context, api_error):
    if context.debug:
        print(f"Submitting OAI compatible completions request...\n")
    expect_api_error = api_error == 'raised'
    completion = await oai_chat_completions(context.prompts.pop(),
                                            context.system_prompt,
                                            context.base_url,
                                            '/v1/chat',
                                            False,
                                            model=context.model if hasattr(context, 'model') else None,

                                            n_predict=context.n_predict
                                            if hasattr(context, 'n_predict') else None,

                                            enable_streaming=context.enable_streaming
                                            if hasattr(context, 'enable_streaming') else None,

                                            seed=await completions_seed(context),

                                            user_api_key=context.user_api_key
                                            if hasattr(context, 'user_api_key') else None,

                                            expect_api_error=expect_api_error)
    context.tasks_result.append(completion)
    if context.debug:
        print(f"Completion response: {completion}")
    if expect_api_error:
        assert completion == 401, f"completion must be an 401 status code: {completion}"

    if context.debug:
        print(f"Completion response: {completion}")


@step(u'a prompt')
def step_a_prompt(context):
    context.prompts.append(context.text)


@step(u'a prompt {prompt}')
def step_a_prompt_prompt(context, prompt):
    context.prompts.append(prompt)


@step(u'concurrent completion requests')
@async_run_until_complete()
async def step_concurrent_completion_requests(context):
    await concurrent_requests(context,
                              request_completion,
                              # prompt is inserted automatically
                              context.base_url,
                              debug=context.debug,
                              prompt_prefix=context.prompt_prefix,
                              prompt_suffix=context.prompt_suffix,
                              n_predict=context.n_predict if hasattr(context, 'n_predict') else None,
                              seed=await completions_seed(context),
                              user_api_key=context.user_api_key if hasattr(context,
                                                                           'user_api_key') else None)


@step(u'concurrent OAI completions requests')
@async_run_until_complete
async def step_oai_chat_completions(context):
    await concurrent_requests(context, oai_chat_completions,
                              # user_prompt is inserted automatically
                              context.system_prompt,
                              context.base_url,
                              '/v1/chat/completions',
                              True,  # async_client
                              model=context.model
                              if hasattr(context, 'model') else None,
                              n_predict=context.n_predict
                              if hasattr(context, 'n_predict') else None,
                              enable_streaming=context.enable_streaming
                              if hasattr(context, 'enable_streaming') else None,
                              seed=await completions_seed(context),
                              user_api_key=context.user_api_key
                              if hasattr(context, 'user_api_key') else None)


@step(u'concurrent OAI completions requests no v1')
@async_run_until_complete
async def step_oai_chat_completions(context):
    await concurrent_requests(context, oai_chat_completions,
                              # user_prompt is inserted automatically
                              context.system_prompt,
                              context.base_url,
                              '/chat/completions',
                              True,  # async_client
                              model=context.model
                              if hasattr(context, 'model') else None,
                              n_predict=context.n_predict
                              if hasattr(context, 'n_predict') else None,
                              enable_streaming=context.enable_streaming
                              if hasattr(context, 'enable_streaming') else None,
                              seed=context.seed
                              if hasattr(context, 'seed') else
                              context.server_seed
                              if hasattr(context, 'server_seed') else None,
                              user_api_key=context.user_api_key
                              if hasattr(context, 'user_api_key') else None)


@step(u'all prompts are predicted')
@async_run_until_complete
async def step_all_prompts_are_predicted(context):
    await all_prompts_are_predicted(context)


@step(u'all prompts are predicted with {n_expected_predicted:d} tokens')
@async_run_until_complete
async def step_all_prompts_are_predicted_with_n_tokens(context, n_expected_predicted):
    await all_prompts_are_predicted(context, n_expected_predicted)


async def all_prompts_are_predicted(context, expected_predicted_n=None):
    n_completions = await gather_tasks_results(context)
    assert n_completions > 0
    for i in range(n_completions):
        assert_n_tokens_predicted(context.tasks_result.pop(), expected_predicted_n=expected_predicted_n)
    assert len(context.concurrent_tasks) == 0, f"{len(context.concurrent_tasks)} pending requests"


@step(u'embeddings are computed for')
@async_run_until_complete
async def step_compute_embedding(context):
    context.embeddings = await request_embedding(context.text, base_url=context.base_url)


@step(u'embeddings are generated')
def step_assert_embeddings(context):
    if len(context.prompts) == 0:
        assert_embeddings(context.embeddings)
    else:
        assert len(context.embeddings) == len(context.prompts), (f"unexpected response:\n"
                                                                 f"context.prompts={context.prompts}\n"
                                                                 f"context.embeddings={context.embeddings}")
        for embedding in context.embeddings:
            context.prompts.pop()
            assert_embeddings(embedding)


@step(u'an OAI compatible embeddings computation request for')
@async_run_until_complete
async def step_oai_compute_embeddings(context):
    context.embeddings = await request_oai_embeddings(context.text,
                                                      base_url=context.base_url,
                                                      user_api_key=context.user_api_key,
                                                      model=context.model)


@step(u'an OAI compatible embeddings computation request for multiple inputs')
@async_run_until_complete
async def step_oai_compute_embeddings_multiple_inputs(context):
    context.embeddings = await request_oai_embeddings(context.prompts,
                                                      base_url=context.base_url,
                                                      user_api_key=context.user_api_key,
                                                      model=context.model)


@step(u'concurrent embedding requests')
@async_run_until_complete()
async def step_concurrent_embedding_requests(context):
    await concurrent_requests(context,
                              request_embedding,
                              # prompt is inserted automatically
                              base_url=context.base_url)


@step(u'concurrent OAI embedding requests')
@async_run_until_complete()
async def step_concurrent_oai_embedding_requests(context):
    await concurrent_requests(context,
                              request_oai_embeddings,
                              # prompt is inserted automatically
                              base_url=context.base_url,
                              async_client=True,
                              model=context.model)


@step(u'all embeddings are generated')
@async_run_until_complete()
async def all_embeddings_are_generated(context):
    n_embedding_requests = await gather_tasks_results(context)
    assert n_embedding_requests > 0
    for i in range(n_embedding_requests):
        assert_embeddings(context.tasks_result.pop())


@step(u'tokenizing')
@async_run_until_complete
async def step_tokenize(context):
    context.tokenized_text = context.text
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/tokenize',
                                json={
                                    "content": context.tokenized_text,
                                }) as response:
            assert response.status == 200
            tokenize_json = await response.json()
            context.tokens = tokenize_json['tokens']


@step(u'tokens can be detokenize')
@async_run_until_complete
async def step_detokenize(context):
    assert len(context.tokens) > 0
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{context.base_url}/detokenize',
                                json={
                                    "tokens": context.tokens,
                                }) as response:
            assert response.status == 200
            detokenize_json = await response.json()
            # SPM tokenizer adds a whitespace prefix: https://github.com/google/sentencepiece/issues/15
            assert context.tokenized_text == detokenize_json['content'].strip()


@step(u'an OPTIONS request is sent from {origin}')
@async_run_until_complete
async def step_options_request(context, origin):
    async with aiohttp.ClientSession() as session:
        async with session.options(f'{context.base_url}/v1/chat/completions',
                                   headers={"Origin": origin}) as response:
            assert response.status == 200
            context.options_response = response


@step(u'CORS header {cors_header} is set to {cors_header_value}')
def step_check_options_header_value(context, cors_header, cors_header_value):
    assert context.options_response.headers[cors_header] == cors_header_value


@step(u'prometheus metrics are exposed')
@async_run_until_complete
async def step_prometheus_metrics_exported(context):
    async with aiohttp.ClientSession() as session:
        async with await session.get(f'{context.base_url}/metrics') as metrics_response:
            assert metrics_response.status == 200
            assert metrics_response.headers['Content-Type'] == "text/plain; version=0.0.4"
            metrics_raw = await metrics_response.text()
            metric_exported = False
            if context.debug:
                print(f"/metrics answer:\n{metrics_raw}\n")
            for metric in parser.text_string_to_metric_families(metrics_raw):
                match metric.name:
                    case "llamacpp:kv_cache_usage_ratio":
                        assert len(metric.samples) > 0
                        metric_exported = True
            assert metric_exported, "No metrics exported"


@step(u'available models')
def step_available_models(context):
    # openai client always expects an api_key
    openai.api_key = context.user_api_key if context.user_api_key is not None else 'nope'
    openai.api_base = f'{context.base_url}/v1'
    context.models = openai.Model.list().data


@step(u'{n_model:d} models are supported')
def step_supported_models(context, n_model):
    if context.debug:
        print("server models available:", context.models)
    assert len(context.models) == n_model


@step(u'model {i_model:d} is {param} {preposition} {param_value}')
def step_supported_models(context, i_model, param, preposition, param_value):
    assert i_model < len(context.models)
    model = context.models[i_model]

    param_value = param_value.split(' ', 1)[0]
    match param:
        case 'identified':
            value = model.id
        case 'trained':
            value = str(model.meta.n_ctx_train)
        case _:
            assert False, "param {param} not supported"
    assert param_value == value, f"model param {param} {value} != {param_value}"


async def concurrent_requests(context, f_completion, *args, **kwargs):
    n_prompts = len(context.prompts)
    if context.debug:
        print(f"starting {n_prompts} concurrent completion requests...")
    assert n_prompts > 0
    for prompt_no in range(n_prompts):
        shifted_args = [context.prompts.pop(), *args]
        context.concurrent_tasks.append(asyncio.create_task(f_completion(*shifted_args, **kwargs)))
    await asyncio.sleep(0.1)


async def request_completion(prompt,
                             base_url,
                             debug=False,
                             prompt_prefix=None,
                             prompt_suffix=None,
                             n_predict=None,
                             seed=None,
                             expect_api_error=None,
                             user_api_key=None):
    if debug:
        print(f"Sending completion request: {prompt}")
    origin = "my.super.domain"
    headers = {
        'Origin': origin
    }
    if user_api_key is not None:
        if debug:
            print(f"Set user_api_key: {user_api_key}")
        headers['Authorization'] = f'Bearer {user_api_key}'

    async with aiohttp.ClientSession() as session:
        async with session.post(f'{base_url}/completion',
                                json={
                                    "input_prefix": prompt_prefix,
                                    "prompt": prompt,
                                    "input_suffix": prompt_suffix,
                                    "n_predict": n_predict if n_predict is not None else -1,
                                    "seed": seed if seed is not None else 42
                                },
                                headers=headers,
                                timeout=3600) as response:
            if expect_api_error is None or not expect_api_error:
                assert response.status == 200
                assert response.headers['Access-Control-Allow-Origin'] == origin
                return await response.json()
            else:
                return response.status


async def oai_chat_completions(user_prompt,
                               system_prompt,
                               base_url,
                               base_path,
                               async_client,
                               debug=False,
                               model=None,
                               n_predict=None,
                               enable_streaming=None,
                               seed=None,
                               user_api_key=None,
                               expect_api_error=None):
    if debug:
        print(f"Sending OAI Chat completions request: {user_prompt}")
    # openai client always expects an api key
    user_api_key = user_api_key if user_api_key is not None else 'nope'
    seed = seed if seed is not None else 42
    enable_streaming = enable_streaming if enable_streaming is not None else False
    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "model": model,
        "max_tokens": n_predict,
        "stream": enable_streaming,
        "seed": seed
    }
    completion_response = {
        'content': '',
        'timings': {
            'predicted_n': 0
        }
    }
    if async_client:
        origin = 'llama.cpp'
        headers = {'Authorization': f'Bearer {user_api_key}', 'Origin': origin}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{base_url}{base_path}',
                                    json=payload,
                                    headers=headers) as response:
                if enable_streaming:
                    assert response.status == 200
                    assert response.headers['Access-Control-Allow-Origin'] == origin
                    assert response.headers['Content-Type'] == "text/event-stream"
                    event_received = True
                    while event_received:
                        event_received = False
                        async for line_in_bytes in response.content:
                            line = line_in_bytes.decode('utf8')
                            line = line.rstrip('\n').rstrip('\r')
                            if line == '':
                                continue
                            event_data = line.split(': ', 1)
                            assert event_data[0] == 'data', f'Bad event code received: ```{event_data}```'
                            chunk_raw = event_data[1]

                            chunk = json.loads(chunk_raw)
                            assert len(chunk['choices']) == 1, f"no choices provided, line ```{line}```"
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta:
                                completion_response['content'] += delta['content']
                                completion_response['timings']['predicted_n'] += 1
                else:
                    if expect_api_error is None or not expect_api_error:
                        assert response.status == 200
                        assert response.headers['Access-Control-Allow-Origin'] == origin
                        assert response.headers['Content-Type'] == "application/json; charset=utf-8"
                        chat_completion_raw = await response.json()
                        completion_response = {
                            'content': chat_completion_raw['choices'][0]['message'],
                            'timings': {
                                'predicted_n': chat_completion_raw['usage']['completion_tokens']
                            }
                        }
                    else:
                        return response.status
    else:
        try:
            openai.api_key = user_api_key
            openai.api_base = f'{base_url}{base_path}'
            chat_completion = openai.Completion.create(
                messages=payload['messages'],
                model=model,
                max_tokens=n_predict,
                stream=enable_streaming,
                seed=seed
            )
        except openai.error.APIError as e:
            if expect_api_error is not None and expect_api_error:
                return 401
            else:
                assert False, f'error raised: {e}'

        if enable_streaming:
            for chunk in chat_completion:
                assert len(chunk.choices) == 1
                delta = chunk.choices[0].delta
                if 'content' in delta:
                    completion_response['content'] += delta['content']
                    completion_response['timings']['predicted_n'] += 1
        else:
            assert len(chat_completion.choices) == 1
            completion_response = {
                'content': chat_completion.choices[0].message.content,
                'timings': {
                    'predicted_n': chat_completion.usage.completion_tokens
                }
            }
    if debug:
        print("OAI response formatted to llama.cpp:", completion_response)
    return completion_response


async def request_embedding(content, base_url=None):
    async with aiohttp.ClientSession() as session:
        async with session.post(f'{base_url}/embedding',
                                json={
                                    "content": content,
                                }) as response:
            assert response.status == 200
            response_json = await response.json()
            return response_json['embedding']


async def request_oai_embeddings(input,
                                 base_url=None, user_api_key=None,
                                 model=None, async_client=False):
    # openai client always expects an api_key
    user_api_key = user_api_key if user_api_key is not None else 'nope'
    if async_client:
        origin = 'llama.cpp'
        if user_api_key is not None:
            headers = {'Authorization': f'Bearer {user_api_key}', 'Origin': origin}
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{base_url}/v1/embeddings',
                                    json={
                                        "input": input,
                                        "model": model,
                                    },
                                    headers=headers) as response:
                assert response.status == 200, f"received status code not expected: {response.status}"
                assert response.headers['Access-Control-Allow-Origin'] == origin
                assert response.headers['Content-Type'] == "application/json; charset=utf-8"
                response_json = await response.json()
                assert response_json['model'] == model, f"invalid model received: {response_json['model']}"
                assert response_json['object'] == 'list'
                return response_json['data']
    else:
        openai.api_key = user_api_key
        openai.api_base = f'{base_url}/v1'
        oai_embeddings = openai.Embedding.create(
            model=model,
            input=input,
        )

        if isinstance(input, collections.abc.Sequence):
            embeddings = []
            for an_oai_embeddings in oai_embeddings.data:
                embeddings.append(an_oai_embeddings.embedding)
        else:
            embeddings = oai_embeddings.data.embedding
        return embeddings


def assert_n_tokens_predicted(completion_response, expected_predicted_n=None, re_content=None):
    content = completion_response['content']
    n_predicted = completion_response['timings']['predicted_n']
    assert len(content) > 0, "no token predicted"
    if re_content is not None:
        p = re.compile(re_content, flags=RegexFlag.IGNORECASE | RegexFlag.MULTILINE | RegexFlag.DOTALL)
        matches = p.finditer(content)
        last_match = 0
        highlighted = ''
        for match in matches:
            start, end = match.span()
            highlighted += content[last_match: start]
            highlighted += '\x1b[33m'
            highlighted += content[start: end]
            highlighted += '\x1b[0m'
            last_match = end
        highlighted += content[last_match:]
        if 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON':
          print(f"Checking completion response: {highlighted}\n")
        assert last_match > 0, f'/{re_content}/ must match ```{highlighted}```'
    if expected_predicted_n and expected_predicted_n > 0:
        assert n_predicted == expected_predicted_n, (f'invalid number of tokens predicted:'
                                                     f' {n_predicted} <> {expected_predicted_n}')



async def gather_tasks_results(context):
    n_tasks = len(context.concurrent_tasks)
    if context.debug:
        print(f"Waiting for all {n_tasks} tasks results...\n")
    for task_no in range(n_tasks):
        context.tasks_result.append(await context.concurrent_tasks.pop())
    n_completions = len(context.tasks_result)
    return n_completions


async def wait_for_health_status(context,
                                 base_url,
                                 expected_http_status_code,
                                 expected_health_status,
                                 timeout=3,
                                 params=None,
                                 slots_idle=None,
                                 slots_processing=None,
                                 expected_slots=None):
    if context.debug:
        print(f"Starting checking for health for expected_health_status={expected_health_status}\n")
    interval = 0.5
    counter = 0
    async with aiohttp.ClientSession() as session:
        while True:
            async with await session.get(f'{base_url}/health', params=params) as health_response:
                status_code = health_response.status
                health = await health_response.json()
                if context.debug:
                    print(f"HEALTH - response for expected health status='{expected_health_status}' on "
                          f"'{base_url}/health'?{params} is {health}\n")
                if (status_code == expected_http_status_code
                        and health['status'] == expected_health_status
                        and (slots_idle is None or health['slots_idle'] == slots_idle)
                        and (slots_processing is None or health['slots_processing'] == slots_processing)):
                    if expected_slots is not None:
                        assert_slots_status(health['slots'], expected_slots)
                    return
                if (status_code == expected_http_status_code
                        and health['status'] == expected_health_status
                        and (slots_idle is None or health['slots_idle'] == slots_idle)
                        and (slots_processing is None or health['slots_processing'] == slots_processing)):
                    if expected_slots is not None:
                        assert_slots_status(health['slots'], expected_slots)
                    return
            await asyncio.sleep(interval)

            counter += interval
            if counter >= timeout:
                # Sometimes health requests are triggered after completions are predicted
                if expected_http_status_code == 503:
                    if len(context.tasks_result) == 0:
                        print("\x1b[5;37;43mWARNING: forcing concurrent tasks,"
                              " busy health check missed, probably too fast inference\x1b[0m\n")
                        n_completions = await gather_tasks_results(context)
                        if n_completions > 0:
                            return

                assert False, f'{expected_health_status} timeout exceeded {counter}s>={timeout}'


def assert_embeddings(embeddings):
    assert len(embeddings) > 0
    embeddings_computed = False
    for emb in embeddings:
        if emb != 0:
            embeddings_computed = True
    assert embeddings_computed, f"Embeddings: {embeddings}"


async def request_slots_status(context, expected_slots):
    async with aiohttp.ClientSession() as session:
        async with await session.get(f'{context.base_url}/slots') as slots_response:
            assert slots_response.status == 200
            slots = await slots_response.json()
            assert_slots_status(slots, expected_slots)


def assert_slots_status(slots, expected_slots):
    assert len(slots) == len(expected_slots)
    for slot_id, (expected, slot) in enumerate(zip(expected_slots, slots)):
        for key in expected:
            assert expected[key] == slot[key], (f"invalid slot {slot_id}"
                                                f" expected[{key}] != slot[{key}]"
                                                f" = {expected[key]} != {slot[key]}")


async def completions_seed(context):
    return context.seed if hasattr(context, 'seed') and context.seed is not None \
        else context.server_seed if hasattr(context, 'server_seed') else None


def start_server_background(context):
    context.server_path = '../../../build/bin/server'
    if 'LLAMA_SERVER_BIN_PATH' in os.environ:
        context.server_path = os.environ['LLAMA_SERVER_BIN_PATH']
    server_args = [
        '--host', context.server_fqdn,
        '--port', context.server_port,
        '--model', context.model_file
    ]
    if context.n_batch:
        server_args.extend(['--batch-size', context.n_batch])
    if context.n_gpu_layer:
        server_args.extend(['--n-gpu-layers', context.n_gpu_layer])
    if context.server_continuous_batching:
        server_args.append('--cont-batching')
    if context.server_embeddings:
        server_args.append('--embedding')
    if context.server_metrics:
        server_args.append('--metrics')
    if context.model_alias:
        server_args.extend(['--alias', context.model_alias])
    if context.n_ctx:
        server_args.extend(['--ctx-size', context.n_ctx])
    if context.n_slots:
        server_args.extend(['--parallel', context.n_slots])
    if context.n_server_predict:
        server_args.extend(['--n-predict', context.n_server_predict])
    if context.server_api_key:
        server_args.extend(['--api-key', context.server_api_key])
    if context.n_ga:
        server_args.extend(['--grp-attn-n', context.n_ga])
    if context.n_ga_w:
        server_args.extend(['--grp-attn-w', context.n_ga_w])
    if context.debug:
        server_args.append('--verbose')
    if 'SERVER_LOG_FORMAT_JSON' not in os.environ:
        server_args.extend(['--log-format', "text"])
    print(f"starting server with: {context.server_path} {server_args}\n")
    context.server_process = subprocess.Popen(
        [str(arg) for arg in [context.server_path, *server_args]],
        close_fds=True)
    print(f"server pid={context.server_process.pid}")
