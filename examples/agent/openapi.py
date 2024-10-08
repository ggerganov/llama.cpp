import aiohttp
import json
import sys
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
                assert body is not None, f'Missing required body parameter: {self.body["name"]}'
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
                if response.status == 500:
                    raise Exception(await response.text())
                response.raise_for_status()
                response_json = await response.json()

        return response_json

async def discover_tools(tool_endpoints: list[str], verbose) -> tuple[dict, list]:
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
                if verbose:
                    print(f'Function {fn.__name__}: params schema: {fn.parameters_schema}', file=sys.stderr)
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
