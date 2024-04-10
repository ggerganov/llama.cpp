from typing import Any, Dict, List, Set, Tuple, Union
import json

from pydantic import Json

class SchemaToTypeScriptConverter:
    # TODO: comments for arguments!
    # // Get the price of a particular car model
    # type get_car_price = (_: {
    # // The name of the car model.
    # car_name: string,
    # }) => any;

    # // get the weather of a location
    # type get_weather = (_: {
    # // where to get weather.
    # location: string,
    # }) => any;

    def __init__(self, allow_fetch: bool = True):
        self._refs: Dict[str, Json[Any]] = {}
        self._refs_being_resolved: Set[str] = set()
        self._allow_fetch = allow_fetch

    def resolve_refs(self, schema: Json[Any], url: str):
        '''
            Resolves all $ref fields in the given schema, fetching any remote schemas,
            replacing $ref with absolute reference URL and populating self._refs with the
            respective referenced (sub)schema dictionaries.
        '''
        def visit(n: Json[Any]):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get('$ref')
                if ref is not None and ref not in self._refs:
                    if ref.startswith('https://'):
                        assert self._allow_fetch, 'Fetching remote schemas is not allowed (use --allow-fetch for force)'
                        import requests

                        frag_split = ref.split('#')
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(requests.get(ref).json(), base_url)
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == '':
                            return target
                    elif ref.startswith('#/'):
                        target = schema
                        ref = f'{url}{ref}'
                        n['$ref'] = ref
                    else:
                        raise ValueError(f'Unsupported ref {ref}')

                    for sel in ref.split('#')[-1].split('/')[1:]:
                        assert target is not None and sel in target, f'Error resolving ref {ref}: {sel} not in {target}'
                        target = target[sel]

                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)

            return n
        return visit(schema)

    def _desc_comment(self, schema: Json[Any]):
        desc = schema.get("description", "").replace("\n", "\n// ") if 'description' in schema else None
        return f'// {desc}\n' if desc else ''

    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], additional_properties: Union[bool, Any]):
        if additional_properties == True:
            additional_properties = {}
        elif additional_properties == False:
            additional_properties = None

        return "{\n" + ',\n'.join([
            f'{self._desc_comment(prop_schema)}{prop_name}{"" if prop_name in required else "?"}: {self.visit(prop_schema)}'
            for prop_name, prop_schema in properties
        ] + (
            [f"{self._desc_comment(additional_properties) if isinstance(additional_properties, dict) else ''}[key: string]: {self.visit(additional_properties)}"]
            if additional_properties is not None else []
        )) + "\n}"

    def visit(self, schema: Json[Any]):
        def print_constant(v):
            return json.dumps(v)

        schema_type = schema.get('type')
        schema_format = schema.get('format')

        if 'oneOf' in schema or 'anyOf' in schema:
            return '|'.join(self.visit(s) for s in schema.get('oneOf') or schema.get('anyOf') or [])

        elif isinstance(schema_type, list):
            return '|'.join(self.visit({'type': t}) for t in schema_type)

        elif 'const' in schema:
            return print_constant(schema['const'])

        elif 'enum' in schema:
            return '|'.join((print_constant(v) for v in schema['enum']))

        elif schema_type in (None, 'object') and \
              ('properties' in schema or \
              ('additionalProperties' in schema and schema['additionalProperties'] is not True)):
            required = set(schema.get('required', []))
            properties = list(schema.get('properties', {}).items())
            return self._build_object_rule(properties, required, schema.get('additionalProperties'))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get('$ref')) is not None:
                    comp_schema = self._refs[ref]

                if 'properties' in comp_schema:
                    for prop_name, prop_schema in comp_schema['properties'].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema['allOf']:
                if 'anyOf' in t:
                    for tt in t['anyOf']:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._build_object_rule(properties, required, additional_properties={})

        elif schema_type in (None, 'array') and ('items' in schema or 'prefixItems' in schema):
            items = schema.get('items') or schema['prefixItems']
            if isinstance(items, list):
                return '[' + ', '.join(self.visit(item) for item in items) + '][]'
            else:
                return self.visit(items) + '[]'

        elif schema_type in (None, 'string') and schema_format == 'date-time':
            return 'Date'

        elif (schema_type == 'object') or (len(schema) == 0):
            return 'any'

        else:
            return 'number' if schema_type == 'integer' else schema_type or 'any'
