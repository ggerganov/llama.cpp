# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp",
#     "fastapi",
#     "pydantic",
#     "typer",
#     "uvicorn",
# ]
# ///
import aiohttp
import asyncio
from functools import wraps
import json
import logging
import os
from pydantic import BaseModel
import sys
import typer
from typing import Optional
import urllib.parse

class OpenAPIMethod:
    def __init__(self, url, name, descriptor, catalog):
        '''
        Wraps a remote OpenAPI method as an async Python function.
        '''
        self.url = url
        self.__name__ = name

        assert 'post' in descriptor, 'Only POST methods are supported'
        post_descriptor = descriptor['post']

        self.__doc__ = post_descriptor.get('description', '')
        parameters = post_descriptor.get('parameters', [])
        request_body = post_descriptor.get('requestBody')

        self.parameters = {p['name']: p for p in parameters}
        assert all(param['in'] == 'query' for param in self.parameters.values()), f'Only query path parameters are supported (path: {url}, descriptor: {json.dumps(descriptor)})'

        self.body = None
        if request_body:
            assert 'application/json' in request_body['content'], f'Only application/json is supported for request body (path: {url}, descriptor: {json.dumps(descriptor)})'

            body_name = 'body'
            i = 2
            while body_name in self.parameters:
                body_name = f'body{i}'
                i += 1

            self.body = dict(
                name=body_name,
                required=request_body['required'],
                schema=request_body['content']['application/json']['schema'],
            )

        self.parameters_schema = dict(
            type='object',
            properties={
                **({
                    self.body['name']: self.body['schema']
                } if self.body else {}),
                **{
                    name: param['schema']
                    for name, param in self.parameters.items()
                }
            },
            required=[name for name, param in self.parameters.items() if param['required']] + ([self.body['name']] if self.body and self.body['required'] else [])
        )

        if (components := catalog.get('components', {})) is not None:
            if (schemas := components.get('schemas')) is not None:
                del schemas['HTTPValidationError']
                del schemas['ValidationError']
                if not schemas:
                    del components['schemas']
            if components:
                self.parameters_schema['components'] = components

    async def __call__(self, **kwargs):
        if self.body:
            body = kwargs.pop(self.body['name'], None)
            if self.body['required']:
                assert body is not None, f'Missing required body parameter: {self.body['name']}'
        else:
            body = None

        query_params = {}
        for name, param in self.parameters.items():
            value = kwargs.pop(name, None)
            if param['required']:
                assert value is not None, f'Missing required parameter: {name}'

            assert param['in'] == 'query', 'Only query parameters are supported'
            query_params[name] = value

        params = '&'.join(f'{name}={urllib.parse.quote(str(value))}' for name, value in query_params.items() if value is not None)
        url = f'{self.url}?{params}'
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body) as response:
                response.raise_for_status()
                response_json = await response.json()

        return response_json

async def discover_tools(tool_endpoints: list[str], logger) -> tuple[dict, list]:
    tool_map = {}
    tools = []

    async with aiohttp.ClientSession() as session:
        for url in tool_endpoints:
            assert url.startswith('http://') or url.startswith('https://'), f'Tools must be URLs, not local files: {url}'

            catalog_url = f'{url}/openapi.json'
            async with session.get(catalog_url) as response:
                response.raise_for_status()
                catalog = await response.json()

            for path, descriptor in catalog['paths'].items():
                fn = OpenAPIMethod(url=f'{url}{path}', name=path.replace('/', ' ').strip().replace(' ', '_'), descriptor=descriptor, catalog=catalog)
                tool_map[fn.__name__] = fn
                logger.debug('Function %s: params schema: %s', fn.__name__, fn.parameters_schema)
                tools.append(dict(
                        type='function',
                        function=dict(
                            name=fn.__name__,
                            description=fn.__doc__ or '',
                            parameters=fn.parameters_schema,
                        )
                    )
                )

    return tool_map, tools


def typer_async_workaround():
    'Adapted from https://github.com/fastapi/typer/issues/950#issuecomment-2351076467'
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    return decorator

@typer_async_workaround()
async def main(
    goal: str,
    model: str = 'gpt-4o',
    tools: Optional[list[str]] = None,
    max_iterations: Optional[int] = 10,
    verbose: bool = False,
    cache_prompt: bool = True,
    seed: Optional[int] = None,
    interactive: bool = True,
    openai: bool = False,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

    if endpoint is None:
        if openai:
            endpoint = 'https://api.openai.com/v1/'
        else:
            endpoint = 'http://localhost:8080/v1/'
    if api_key is None:
        if openai:
            api_key = os.environ.get('OPENAI_API_KEY')

    tool_map, tools = await discover_tools(tools or [], logger=logger)

    sys.stdout.write(f'🛠️  Tools: {", ".join(tool_map.keys()) if tool_map else "<none>"}\n')

    messages = [
        dict(
            role='user',
            content=goal,
        )
    ]

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    async def run_turn():
        for i in range(max_iterations or sys.maxsize):
            url = f'{endpoint}chat/completions'
            payload = dict(
                messages=messages,
                model=model,
                tools=tools,
            )
            if not openai:
                payload.update(dict(
                    seed=seed,
                    cache_prompt=cache_prompt,
                )) # type: ignore

            logger.debug('Calling %s with %s', url, json.dumps(payload, indent=2))
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(url, json=payload) as response:
                    logger.debug('Response: %s', response)
                    response.raise_for_status()
                    response = await response.json()

            assert len(response['choices']) == 1
            choice = response['choices'][0]

            content = choice['message']['content']
            if choice['finish_reason'] == 'tool_calls':
                messages.append(choice['message'])
                assert choice['message']['tool_calls']
                for tool_call in choice['message']['tool_calls']:
                    if content:
                        print(f'💭 {content}')

                    name = tool_call['function']['name']
                    args = json.loads(tool_call['function']['arguments'])
                    pretty_call = f'{name}({", ".join(f"{k}={v.model_dump_json() if isinstance(v, BaseModel) else json.dumps(v)}" for k, v in args.items())})'
                    logger.info(f'⚙️  {pretty_call}')
                    sys.stdout.flush()
                    tool_result = await tool_map[name](**args)
                    tool_result_str = json.dumps(tool_result)
                    logger.info(' → %d chars', len(tool_result_str))
                    logger.debug('%s', tool_result_str)
                    messages.append(dict(
                        tool_call_id=tool_call.get('id'),
                        role='tool',
                        content=tool_result_str,
                    ))
            else:
                assert content
                print(content)
                return

        if max_iterations is not None:
            raise Exception(f'Failed to get a valid response after {max_iterations} tool calls')

    while interactive:
        await run_turn()
        messages.append(dict(
            role='user',
            content=input('💬 ')
        ))


if __name__ == '__main__':
    typer.run(main)
